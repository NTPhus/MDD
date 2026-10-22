r"""Probe actual local Qwen conversation/completion behavior without audio or APIs.

Usage: .venv\Scripts\python.exe -u scripts/check_conversation_context_live.py
Use --natural-only for three dialogue probes covering a topic change, unclear
speech, and a simple thank-you. The default performs the original four probes.
Use --echo-only for four conversation probes focused on restatement and context.
All generations are CPU-only. This does not train, call a network API,
or measure pronunciation quality. Raw replies are saved for qualitative review;
passing policy checks are not evidence that the conversation sounds natural.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import local_chat_process
from scripts.local_speech_engine import conversation_history, dialogue_request, follow_up
from scripts.sentence_correction import correction_request, parse_correction, _words


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/generated/conversation-continuity-qa/live-summary.json")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--natural-only", action="store_true",
                       help="Run three natural-conversation probes, without sentence completion.")
    modes.add_argument("--echo-only", action="store_true",
                       help="Run four dialogue probes for short/full utterances and recent context.")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    summary = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model_id": "qwen-base", "model": "Qwen3-0.6B", "device": "cpu", "adapter": None,
        "training": False, "network_api_calls": False, "audio_inference": False,
        "mode": "echo_only" if args.echo_only else "natural_only" if args.natural_only else "continuity",
        "generations_planned": 3 if args.natural_only else 4,
        "generations": [], "policy_checks": [],
        "qualitative_review_required": True,
        "limitations": [
            "These probes exercise real local text generations, not ASR, phoneme analysis, microphone or TTS.",
            "History delivery and parser policy checks do not establish conversational quality or factual correctness.",
            "An unsafe or incomplete sentence suggestion is withheld; unchanged speech remains usable.",
            "Conversation sampling is stochastic with no fixed seed; one reply per case does not establish reliability.",
        ],
    }

    def save():
        summary["elapsed_sec"] = round(time.monotonic() - started, 3)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def check(name, passed, detail):
        summary["policy_checks"].append({"name": name, "passed": bool(passed), "detail": detail})

    def generate(label, request):
        print(f"[{len(summary['generations']) + 1}/{summary['generations_planned']}] {label}: starting local CPU generation", flush=True)
        chunks = []
        run_started = time.monotonic()
        metadata = local_chat_process.complete(
            request["messages"], {"device": "cpu", "adapter": None,
                                  "generation_profile": request.get("generation_profile")},
            request["max_tokens"], request["timeout"],
            on_text=chunks.append, cancelled=threading.Event(),
            on_ready=lambda: print(f"  {label}: model ready", flush=True),
        )
        record = {"label": label, "messages": request["messages"], "raw_output": "".join(chunks),
                  "generation_profile": request.get("generation_profile"),
                  "max_tokens": request["max_tokens"], "timeout_sec": request["timeout"],
                  "metadata": metadata, "elapsed_sec": round(time.monotonic() - run_started, 3)}
        summary["generations"].append(record)
        save()
        print(f"  {label}: {metadata.get('finish_reason')} in {record['elapsed_sec']}s; "
              f"output={record['raw_output'][:260]!r}", flush=True)
        return record

    def completion(label, transcript, history):
        record = generate(label, correction_request(transcript, history))
        record["transcript"] = transcript
        record["conversation"] = history
        try:
            accepted = parse_correction(record["raw_output"],
                                        record["metadata"].get("finish_reason") == "stop", transcript, history)
        except (ValueError, TypeError) as error:
            record.update(disposition="withheld", reason=str(error), displayed_original=transcript)
        else:
            record.update(disposition="accepted", parsed=accepted)
        print(f"  {label}: suggestion {record['disposition']}", flush=True)
        return record

    def continuity_probes():
        original = "Yesterday I go to the park with my friend"
        grammar = completion("complete_utterance_no_grammar_rewrite", original, [])
        unchanged = grammar["disposition"] == "withheld" or _words(grammar["parsed"]["corrected"]) == _words(original)
        check("complete_utterance_words_preserved_or_suggestion_withheld", unchanged,
              "A grammar rewrite must not be published, including changing go to went.")
        save()

        context_request = {"model_id": "qwen-trained", "topic_id": "travel", "history": []}
        opener_context = conversation_history(context_request)
        japan = completion("fragment_with_authored_travel_opener", "Japan", opener_context)
        payload = json.loads(japan["messages"][-1]["content"])
        check("completion_receives_actual_topic_opener", payload["conversation"] == opener_context,
              "The exact authored assistant question is supplied as contextual data.")
        if japan["disposition"] == "accepted":
            check("accepted_fragment_uses_completion_policy", japan["parsed"].get("mode") == "completion_only",
                  "Production parser accepted the suggestion under its word-preserving contextual completion rules.")
        else:
            check("unusable_fragment_suggestion_is_withheld", "parsed" not in japan,
                  "A rejected model suggestion is recorded without presenting it as a completed sentence.")
        save()

        first_request = dialogue_request(context_request, "Japan")
        first = generate("dialogue_first_turn_japan", first_request)
        first_hint, first_source = follow_up(first["raw_output"], "travel", 1,
                                             transcript="Japan", history=opener_context)
        first_reply = first["raw_output"]
        first.update(effective_reply=first_reply, follow_up={"source": first_source, **first_hint})
        check("first_dialogue_uses_raw_fragment_and_opener",
              first_request["messages"][1:-1] == opener_context
              and first_request["messages"][-1] == {"role": "user", "content": "Japan"},
              "Dialogue receives what the learner actually said, not the completion suggestion.")
        check("first_reply_contains_model_output", bool(first["raw_output"].strip()),
              "The next turn must have a real generated response rather than only a catalog/fallback question.")
        save()

        actual_history = [{"role": "user", "content": "Japan"},
                          {"role": "assistant", "content": first_reply}]
        next_transcript = "I want to try local food with my sister."
        next_request = dialogue_request({**context_request, "history": actual_history}, next_transcript)
        second = generate("dialogue_follow_up_with_actual_previous_reply", next_request)
        second_hint, second_source = follow_up(second["raw_output"], "travel", 2,
                                               transcript=next_transcript,
                                               history=conversation_history({**context_request, "history": actual_history}))
        second_reply = second["raw_output"]
        second.update(effective_reply=second_reply, follow_up={"source": second_source, **second_hint})
        check("second_turn_receives_exact_generated_previous_exchange",
              next_request["messages"][1:-1] == [*opener_context, *actual_history],
              "The actual first reply is handed back to Qwen without an authored follow-up question.")
        check("second_turn_receives_current_raw_utterance",
              next_request["messages"][-1] == {"role": "user", "content": next_transcript},
              "The current user words remain untouched in the dialogue input.")
        check("dialogue_always_uses_base_qwen", first_request["model_id"] == next_request["model_id"] == "qwen-base",
              "The pronunciation adapter is not substituted for the conversation model.")
        check("second_reply_contains_model_output", bool(second["raw_output"].strip()),
              "A nonempty response is required, but does not establish coherence, relevance, or naturalness.")
        check("dialogue_does_not_append_authored_questions",
              first_reply == first["raw_output"] and second_reply == second["raw_output"],
              "A statement, yes/no question, or clarification is retained without a scripted extra question.")

    def natural_probes(cases=None):
        cases = cases or [
            {"label": "food_topic_learner_changes_to_late_train", "topic_id": "food",
             "history": [{"role": "user", "content": "An iced coffee, please."},
                         {"role": "assistant", "content": "Sure, an iced coffee it is."}],
             "transcript": "Actually, my train is late and I don't know how I'll get home.",
             "review": "Does the reply respond to the late train and the learner's concern, rather than return to food or invent their plans?"},
            {"label": "unclear_misheard_fragment", "topic_id": "food", "history": [],
             "transcript": "fravorit... tomorrow...",
             "review": "Does the reply acknowledge uncertainty and ask a brief clarification, instead of guessing an order or preference?"},
            {"label": "normal_thank_you_without_required_question", "topic_id": None,
             "history": [{"role": "user", "content": "My train is late."},
                         {"role": "assistant", "content": "You could check the station board for the next departure."}],
             "transcript": "Thanks, that helps.",
             "review": "Is the reply a natural acknowledgement? A further question is optional; inspect whether it feels necessary rather than scripted."},
        ]
        for case in cases:
            context_history = list(case["history"])
            if case.get("history_from_label"):
                previous = next(record for record in summary["generations"]
                                if record["label"] == case["history_from_label"])
                context_history += [{"role": "user", "content": previous["transcript"]},
                                    {"role": "assistant", "content": previous["raw_output"]}]
            context = {"model_id": "qwen-trained", "topic_id": case["topic_id"], "history": context_history}
            history = conversation_history(context)
            request = dialogue_request(context, case["transcript"])
            record = generate(case["label"], request)
            hint, source = follow_up(record["raw_output"], case["topic_id"], 2,
                                     transcript=case["transcript"], history=history)
            record.update(transcript=case["transcript"], conversation=history,
                          effective_reply=record["raw_output"], follow_up={"source": source, **hint},
                          qualitative_review={"status": "pending", "look_for": case["review"]})
            if case.get("history_from_label"):
                record["history_source_label"] = case["history_from_label"]
            prefix = case["label"]
            check(prefix + "_receives_original_utterance_and_history",
                  request["messages"][1:-1] == history
                  and request["messages"][-1] == {"role": "user", "content": case["transcript"]},
                  "The exact conversation and speech transcript are supplied without completing or rewriting the user message.")
            check(prefix + "_uses_base_qwen", request["model_id"] == "qwen-base",
                  "The conversation uses base Qwen, not the pronunciation coach adapter.")
            check(prefix + "_uses_conversation_sampling",
                  record["metadata"].get("generation_profile") == "conversation"
                  and record["metadata"].get("sampling") == {"do_sample": True, "temperature": .7,
                                                            "top_p": .8, "top_k": 20, "min_p": 0.},
                  "The worker confirms the internal non-thinking conversation profile; structured tasks keep their default decoder.")
            check(prefix + "_no_authored_follow_up", record["effective_reply"] == record["raw_output"],
                  "The application does not append a question; response quality remains a separate qualitative review.")
            check(prefix + "_contains_model_output", bool(record["raw_output"].strip()),
                  "A nonempty response is required; this does not establish relevance or naturalness.")
            if source == "none":
                check(prefix + "_statement_has_no_question_metadata", hint["question"] is None,
                      "A generated statement does not produce a fabricated follow-up question in metadata.")
            save()

    def echo_probes():
        natural_probes([
            {"label": "short_food_preference", "topic_id": "food", "history": [],
             "transcript": "I like pizza.",
             "review": "Does the reply add a new relevant reaction or question without repeating or paraphrasing the learner's pizza preference?"},
            {"label": "full_coffee_story", "topic_id": "daily-life",
             "history": [{"role": "user", "content": "I have the day off today."},
                         {"role": "assistant", "content": "How did you start your day?"}],
             "transcript": "I had coffee with my sister this morning.",
             "review": "Does the reply move the exchange forward without narrating or restating the coffee/sister sentence?"},
            {"label": "most_recent_context_about_sister", "topic_id": "daily-life",
             "history": [{"role": "user", "content": "I have the day off today."},
                         {"role": "assistant", "content": "How did you start your day?"}],
             "history_from_label": "full_coffee_story",
             "transcript": "We had not seen each other for three months.",
             "review": "Does Qwen link we to the recent sister/coffee exchange and respond to the reunion without restating the new sentence?"},
            {"label": "food_topic_changes_to_future_travel_plan", "topic_id": "food", "history": [],
             "history_from_label": "short_food_preference",
             "transcript": "Actually, I am going to Da Nang next month.",
             "review": "Does Qwen follow the travel plan instead of staying with pizza, repeating the travel sentence, or inventing unsupported travel details?"},
        ])

    try:
        if args.echo_only:
            echo_probes()
        elif args.natural_only:
            natural_probes()
        else:
            continuity_probes()
        summary["policy_passed"] = all(item["passed"] for item in summary["policy_checks"])
        summary["status"] = "completed"
    except Exception as error:
        summary.update(status="error", policy_passed=False,
                       error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
        local_chat_process.shutdown()
        print(f"Report: {output}", flush=True)
    print(f"Policy checks: {'PASS' if summary['policy_passed'] else 'FAIL'}; raw conversation requires review.", flush=True)
    return 0 if summary["policy_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
