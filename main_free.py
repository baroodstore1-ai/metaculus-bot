"""
main_free.py - $0 variant of the Metaculus Fall 2026 template bot.

Same forecasting logic as main.py (FallTemplateBot2026), but:
  * LLMs are picked automatically from whatever keys exist, best first:
      1. OPENROUTER_API_KEY  (Metaculus-donated credits once granted; model via FORECAST_MODEL)
      2. GROQ_API_KEY        (free tier, no card: openai/gpt-oss-120b, 1K req/day, 8K tok/min)
      3. GEMINI_API_KEY      (Google AI Studio free tier: flash-lite ~500 req/day)
      4. OPENROUTER_FREE_KEY (personal OpenRouter key, ":free" models, 50 req/day); a personal key
         stored as OPENROUTER_API_KEY without credits is detected by a probe and moved here
      5. GITHUB_TOKEN -> GitHub Models (retired by GitHub 2026-07-30; used only if a probe succeeds)
      6. METACULUS_TOKEN -> Metaculus LLM proxy (only if Metaculus enabled it; probed)
      7. Pollinations anonymous (no key, gpt-oss-20b) - always-available last resort
  * Research: AskNews (free for registered bot makers) if ASKNEWS_* is set,
    otherwise free web research = DuckDuckGo news + Wikipedia, no key needed.
  * Fewer LLM calls per question on free tiers, plus pacing to respect per-minute limits.

Fully autonomous: no human input anywhere. Run:
  python main_free.py --mode test_questions   # bot-testing-area (safe to resubmit)
  python main_free.py --mode tournament       # Fall 2026 FutureEval + MiniBench
  python main_free.py --dry-run-sample        # offline-ish sample question, publishes nothing
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from datetime import datetime, timezone

import dotenv
import requests

dotenv.load_dotenv()

from bot_helpers import print_run_summary_banner, silence_noisy_dependencies  # noqa: E402

silence_noisy_dependencies()

from forecasting_tools import (  # noqa: E402
    AskNewsSearcher,
    BinaryQuestion,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
)

from main import FallTemplateBot2026  # noqa: E402

logger = logging.getLogger("main_free")


def _real(name: str) -> bool:
    v = os.getenv(name, "").strip()
    return bool(v) and v != "REPLACE_ME"


# ----------------------------------------------------------------------------- LLMs
class PacedLlm(GeneralLlm):
    """GeneralLlm that waits a minimum interval between calls (per-minute free-tier limits)."""

    _locks: dict[str, asyncio.Lock] = {}
    _last: dict[str, float] = {}

    def __init__(self, *args, min_interval_s: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._min_interval_s = min_interval_s

    async def invoke(self, prompt, *args, **kwargs):  # type: ignore[override]
        if self._min_interval_s > 0:
            key = f"{id(asyncio.get_running_loop())}:{self.model}"
            lock = PacedLlm._locks.setdefault(key, asyncio.Lock())
            async with lock:
                wait = PacedLlm._last.get(self.model, 0) + self._min_interval_s - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                PacedLlm._last[self.model] = time.monotonic()
        return await super().invoke(prompt, *args, **kwargs)


class FallbackLlm(GeneralLlm):
    """Try the primary LLM; on any error (e.g. free-tier 429 / daily cap) answer with the backup instead."""

    def __init__(self, primary: GeneralLlm, backup: GeneralLlm):
        super().__init__(model=primary.model)
        self.primary, self.backup = primary, backup

    async def invoke(self, prompt, *args, **kwargs):  # type: ignore[override]
        try:
            return await self.primary.invoke(prompt, *args, **kwargs)
        except Exception as e:
            logger.warning(f"{self.primary.model} failed ({type(e).__name__}: {str(e)[:160]}); "
                           f"using backup {self.backup.model}")
            return await self.backup.invoke(prompt, *args, **kwargs)


OPENROUTER_FREE_PREFERENCE = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "openai/gpt-oss-120b:free",
    "deepseek/deepseek-chat-v3.1:free",
]


def _openrouter_key_info(key: str) -> dict:
    """Free metadata call (does not count against the request cap)."""
    try:
        return requests.get("https://openrouter.ai/api/v1/key", timeout=30,
                            headers={"Authorization": f"Bearer {key}"}).json().get("data", {}) or {}
    except Exception:
        return {}


def _openrouter_free_model() -> str | None:
    """Pick the best currently-listed ':free' model (model list is free, no request quota used)."""
    if os.getenv("FREE_FORECAST_MODEL"):
        return os.environ["FREE_FORECAST_MODEL"]
    try:
        ids = {m["id"] for m in requests.get("https://openrouter.ai/api/v1/models", timeout=30).json()["data"]}
    except Exception:
        return OPENROUTER_FREE_PREFERENCE[0]
    for m in OPENROUTER_FREE_PREFERENCE:
        if m in ids:
            return m
    return None


def _pollinations(temp: float, timeout: int = 240) -> PacedLlm:
    # anonymous tier: "1 reply every 30 seconds per IP" -> pace all Pollinations calls 35 s apart
    return _openai_compat("openai", POLLINATIONS_BASE, "anonymous", interval=35, temp=temp,
                          timeout=timeout, tries=5)


GITHUB_MODELS_BASE = "https://models.github.ai/inference"
METACULUS_PROXY_BASE = "https://llm-proxy.metaculus.com/proxy/openai/v1"
POLLINATIONS_BASE = "https://text.pollinations.ai/openai"


def _probe(base: str, model: str, headers: dict) -> bool:
    """Send one tiny chat request; True only if a real OpenAI-style completion comes back."""
    try:
        r = requests.post(base.rstrip("/") + "/chat/completions", timeout=60,
                          headers={"Content-Type": "application/json", **headers},
                          json={"model": model, "max_tokens": 20,
                                "messages": [{"role": "user", "content": "What is 17*3? Reply with the number only."}]})
        content = r.json()["choices"][0]["message"]["content"]
        ok = r.status_code == 200 and bool(content and content.strip())
        logger.info(f"probe {base} {model}: HTTP {r.status_code} -> {'OK' if ok else 'empty'}")
        return ok
    except Exception as e:
        logger.info(f"probe {base} {model}: failed ({type(e).__name__}: {str(e)[:120]})")
        return False


def _openai_compat(model: str, base: str, key: str, *, interval: float, temp: float,
                   timeout: int = 180, tries: int = 4, **kw) -> PacedLlm:
    return PacedLlm(model=f"openai/{model}", base_url=base, api_key=key, temperature=temp,
                    timeout=timeout, allowed_tries=tries, min_interval_s=interval, **kw)


def pick_llms() -> tuple[str, dict]:
    """Return (tier_name, llms dict) for the best provider that actually answers right now."""
    if _real("OPENROUTER_API_KEY"):
        key = os.environ["OPENROUTER_API_KEY"].strip()
        main_m = os.getenv("FORECAST_MODEL", "openrouter/openai/gpt-5.2")
        parse_m = os.getenv("PARSER_MODEL", "openrouter/openai/gpt-4o-mini")
        info = _openrouter_key_info(key)
        remaining = info.get("limit_remaining")
        has_credits = (not info.get("is_free_tier", True)) and (remaining is None or remaining > 1)
        logger.info(f"OpenRouter key: free_tier={info.get('is_free_tier')} limit_remaining={remaining}")
        if has_credits:
            return "openrouter-credits", {
                "default": GeneralLlm(model=main_m, temperature=0.3, timeout=180, allowed_tries=3),
                "summarizer": parse_m,
                "parser": GeneralLlm(model=parse_m, temperature=0.0, timeout=60, allowed_tries=3),
                "researcher": "free-web",
            }
        if not _real("OPENROUTER_FREE_KEY"):  # personal key without credits -> :free models
            # (GitHub Actions passes unset secrets as "", so setdefault() would never fire)
            os.environ["OPENROUTER_FREE_KEY"] = key
    if _real("GROQ_API_KEY"):
        main_m = os.getenv("FORECAST_MODEL", "groq/openai/gpt-oss-120b")
        parse_m = os.getenv("PARSER_MODEL", "groq/llama-3.1-8b-instant")
        return "groq-free", {
            # 8K tokens/min on gpt-oss-120b -> ~1 big call every 40 s
            "default": PacedLlm(model=main_m, temperature=0.3, timeout=120, allowed_tries=4,
                                max_tokens=2500, min_interval_s=40),
            "summarizer": parse_m,
            "parser": PacedLlm(model=parse_m, temperature=0.0, timeout=60, allowed_tries=4,
                               min_interval_s=12),
            "researcher": "free-web",
        }
    if _real("GEMINI_API_KEY"):
        main_m = os.getenv("FORECAST_MODEL", "gemini/gemini-flash-lite-latest")
        return "gemini-free", {
            "default": PacedLlm(model=main_m, temperature=0.3, timeout=120, allowed_tries=4,
                                min_interval_s=8),
            "summarizer": main_m,
            "parser": PacedLlm(model=main_m, temperature=0.0, timeout=60, allowed_tries=4,
                               min_interval_s=8),
            "researcher": "free-web",
        }
    if _real("OPENROUTER_FREE_KEY"):
        key = os.environ["OPENROUTER_FREE_KEY"].strip()
        os.environ["OPENROUTER_API_KEY"] = key
        m = _openrouter_free_model()
        if m:
            full = "openrouter/" + m
            # ~50 free requests/day: spend them on the forecasts only; parsing/summaries go to Pollinations.
            return f"openrouter-free:{m}", {
                "default": FallbackLlm(
                    PacedLlm(model=full, temperature=0.3, timeout=180, allowed_tries=2, min_interval_s=4),
                    _pollinations(0.3)),
                "summarizer": _pollinations(0.3, 120),
                "parser": _pollinations(0.0, 120),
                "researcher": "free-web",
            }
    # GitHub Models (free with the Actions GITHUB_TOKEN; GitHub announced its retirement on 2026-07-30,
    # so this is only used if the probe gets a real completion back).
    gh_tok = os.getenv("GITHUB_TOKEN", "").strip()
    if gh_tok:
        auth = {"Authorization": f"Bearer {gh_tok}"}
        main_m = os.getenv("GITHUB_FORECAST_MODEL", "openai/gpt-4.1")
        parse_m = os.getenv("GITHUB_PARSER_MODEL", "openai/gpt-4.1-mini")
        if _probe(GITHUB_MODELS_BASE, main_m, auth):
            return "github-models", {
                "default": _openai_compat(main_m, GITHUB_MODELS_BASE, gh_tok, interval=7, temp=0.3,
                                          max_tokens=3000),
                "summarizer": _openai_compat(parse_m, GITHUB_MODELS_BASE, gh_tok, interval=5, temp=0.3, timeout=60),
                "parser": _openai_compat(parse_m, GITHUB_MODELS_BASE, gh_tok, interval=5, temp=0.0, timeout=60),
                "researcher": "free-web",
            }
    # Metaculus LLM proxy (only works if Metaculus enabled it for this bot token)
    tok = os.getenv("METACULUS_TOKEN", "").strip()
    if tok and _probe(METACULUS_PROXY_BASE, "gpt-4o", {"Authorization": f"Token {tok}"}):
        return "metaculus-proxy", {
            "default": GeneralLlm(model="metaculus/gpt-4o", temperature=0.3, timeout=120),
            "summarizer": "metaculus/gpt-4o-mini",
            "parser": GeneralLlm(model="metaculus/gpt-4o-mini", temperature=0.0),
            "researcher": "free-web",
        }
    # Last resort: Pollinations anonymous tier (no key, gpt-oss-20b). Real but small model; paced politely.
    return "pollinations-anonymous", {
        "default": _pollinations(0.3),
        "summarizer": _pollinations(0.3, 120),
        "parser": _pollinations(0.0, 120),
        "researcher": "free-web",
    }


# ----------------------------------------------------------------------------- free research
UA = {"User-Agent": "metaculus-forecast-bot/1.0 (research for Metaculus FutureEval)"}


def _ddg_news(query: str, n: int = 8) -> list[str]:
    try:
        from ddgs import DDGS
    except ImportError:
        return []
    out = []
    try:
        with DDGS() as d:
            for r in d.news(query, max_results=n):
                out.append(f"- [{(r.get('date') or '')[:10]}] {r.get('title','')} ({r.get('source','')}): "
                           f"{(r.get('body') or '')[:300]}  <{r.get('url','')}>")
    except Exception as e:  # rate limits etc.
        logger.warning(f"DDG news failed: {e}")
    if not out:
        try:
            with DDGS() as d:
                for r in d.text(query, max_results=n):
                    out.append(f"- {r.get('title','')}: {(r.get('body') or '')[:300]}  <{r.get('href','')}>")
        except Exception as e:
            logger.warning(f"DDG text failed: {e}")
    return out


def _wikipedia(query: str, n: int = 2) -> list[str]:
    out = []
    try:
        s = requests.get("https://en.wikipedia.org/w/api.php", headers=UA, timeout=20, params={
            "action": "query", "list": "search", "srsearch": query, "srlimit": n, "format": "json"}).json()
        for hit in s.get("query", {}).get("search", []):
            t = hit["title"]
            p = requests.get("https://en.wikipedia.org/api/rest_v1/page/summary/" + t.replace(" ", "_"),
                             headers=UA, timeout=20).json()
            out.append(f"- Wikipedia '{t}': {(p.get('extract') or '')[:700]}")
    except Exception as e:
        logger.warning(f"Wikipedia failed: {e}")
    return out


def _queries(q: MetaculusQuestion) -> list[str]:
    text = q.question_text.strip().rstrip("?")
    for pre in ("Will ", "What will ", "How many ", "Which ", "Who will ", "When will "):
        if text.startswith(pre):
            text = text[len(pre):]
            break
    short = " ".join(text.split()[:14])
    return [short, short + " latest news " + datetime.now().strftime("%Y")]


async def free_web_research(q: MetaculusQuestion, max_chars: int = 9000) -> str:
    qs = _queries(q)
    news, wiki = await asyncio.gather(
        asyncio.to_thread(lambda: _ddg_news(qs[0]) + _ddg_news(qs[1], 5)),
        asyncio.to_thread(_wikipedia, qs[0]),
    )
    seen, lines = set(), []
    for line in news:
        k = line[:80]
        if k not in seen:
            seen.add(k)
            lines.append(line)
    body = (f"Free web research gathered {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC "
            f"for queries {qs}.\n\nRecent news/search results:\n" + ("\n".join(lines) or "(none found)")
            + "\n\nBackground:\n" + ("\n".join(wiki) or "(none found)"))
    return body[:max_chars]


# ----------------------------------------------------------------------------- bot
class FreeTierBot(FallTemplateBot2026):
    _structure_output_validation_samples = 1

    async def _binary_prompt_to_forecast(self, question, prompt):
        """Read the final "Probability: ZZ%" line directly; only call the LLM parser if it is missing.
        Saves a free-tier request per sample and avoids weak-parser failures."""
        import re
        from forecasting_tools import ReasonedPrediction
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        hits = re.findall(r"Probability\W{0,3}\s*:?\s*\**\s*(\d{1,3}(?:\.\d+)?)\s*%", reasoning or "", re.I)
        if hits:
            p = max(0.01, min(0.99, float(hits[-1]) / 100))
            logger.info(f"Forecasted URL {question.page_url} with prediction: {p} (regex parse).")
            return ReasonedPrediction(prediction_value=p, reasoning=reasoning)
        return await self._parse_binary_with_llm(question, reasoning)

    async def _parse_binary_with_llm(self, question, reasoning):
        from forecasting_tools import BinaryPrediction, ReasonedPrediction, structure_output
        pred = await structure_output(reasoning, BinaryPrediction, model=self.get_llm("parser", "llm"),
                                      num_validation_samples=1)
        p = max(0.01, min(0.99, pred.prediction_in_decimal))
        return ReasonedPrediction(prediction_value=p, reasoning=reasoning)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            if _real("ASKNEWS_API_KEY") or (_real("ASKNEWS_CLIENT_ID") and _real("ASKNEWS_SECRET")):
                try:
                    research = await AskNewsSearcher().call_preconfigured_version(
                        "asknews/news-summaries", question.question_text)
                except Exception as e:
                    logger.warning(f"AskNews failed, falling back to free web research: {e}")
            if not research:
                research = await free_web_research(question)
            logger.info(f"Research for {question.page_url}:\n{research[:1500]}")
            return research


def build_bot(publish: bool) -> tuple[str, FreeTierBot]:
    tier, llms = pick_llms()
    paid = tier in ("openrouter-credits", "github-models", "metaculus-proxy")
    bot = FreeTierBot(
        research_reports_per_question=1,
        predictions_per_research_report=5 if paid else 3,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=os.getenv("REPORTS_DIR") or None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms=llms,
    )
    return tier, bot


SAMPLE = dict(
    question_text="Will the US Federal Reserve cut the federal funds target range at its December 2026 FOMC meeting?",
    background_info="The FOMC meets eight times a year; its December 2026 meeting concludes mid-December.",
    resolution_criteria="Resolves Yes if the FOMC statement at the conclusion of the December 2026 meeting announces "
                        "a lower federal funds target range than before the meeting.",
    fine_print="",
    page_url="https://www.metaculus.com/questions/0/sample-dry-run/",
    id_of_post=0,
    id_of_question=0,
)


async def dry_run_sample(mock_llm: bool) -> None:
    q = BinaryQuestion(**SAMPLE)
    research = await free_web_research(q)
    print("=== RESEARCH (free web) ===\n" + research[:3000] + "\n")
    tier, bot = build_bot(publish=False)
    print(f"=== LLM tier selected: {tier} ===")
    if mock_llm:
        canned = ("(a) ~2 months. (b) Status quo: no change. (c) Inflation sticky -> hold. "
                  "(d) Weak labor data -> cut.\nProbability: 41%")

        async def fake_invoke(self, prompt, *a, **k):
            if "structure" in str(prompt).lower() or "json" in str(prompt).lower():
                return '{"prediction_in_decimal": 0.41}'
            return canned
        GeneralLlm.invoke = fake_invoke  # type: ignore[assignment]
        print("(mock LLM: no API key on this machine, exercising the pipeline only)")
    report = await bot.forecast_question(q, return_exceptions=True)
    if isinstance(report, BaseException):
        print("Pipeline error:", repr(report))
    else:
        print("=== FORECAST ===", report.prediction)
        print(report.explanation[:1500])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tournament", "metaculus_cup", "test_questions"], default="tournament")
    ap.add_argument("--dry-run-sample", action="store_true")
    ap.add_argument("--mock-llm", action="store_true")
    ap.add_argument("--no-publish", action="store_true")
    args = ap.parse_args()

    if args.dry_run_sample:
        asyncio.run(dry_run_sample(args.mock_llm))
        raise SystemExit(0)

    if not _real("METACULUS_TOKEN"):
        raise SystemExit("METACULUS_TOKEN missing. Create a bot under Metaculus Settings -> My Forecasting Bots.")
    publish = not args.no_publish
    tier, bot = build_bot(publish)
    print(f"🤖 mode={args.mode} tier={tier} publish={publish}")
    client = MetaculusClient()
    if args.mode == "tournament":
        reports = asyncio.run(bot.forecast_on_tournament(client.CURRENT_AI_COMPETITION_ID, return_exceptions=True))
        reports += asyncio.run(bot.forecast_on_tournament(client.CURRENT_MINIBENCH_ID, return_exceptions=True))
        url = "https://www.metaculus.com/tournament/fall-futureeval-2026/"
    elif args.mode == "metaculus_cup":
        bot.skip_previously_forecasted_questions = False
        reports = asyncio.run(bot.forecast_on_tournament(client.CURRENT_METACULUS_CUP_ID, return_exceptions=True))
        url = "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/"
    else:
        bot.skip_previously_forecasted_questions = False
        reports = asyncio.run(bot.forecast_on_tournament("bot-testing-area", return_exceptions=True))
        url = "https://www.metaculus.com/tournament/bot-testing-area/"
    ok = [r for r in reports if not isinstance(r, BaseException)]
    try:
        bot.log_report_summary(reports)
    except RuntimeError as e:  # some questions failed; only fail the job if none succeeded
        logger.error(f"{len(reports) - len(ok)} of {len(reports)} questions failed: {str(e)[:500]}")
        if reports and not ok:
            raise
    print_run_summary_banner(reports, will_publish=publish, tournament_url=url)
