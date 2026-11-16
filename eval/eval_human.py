"""
Eval 5: Human Rating Framework
Generates blinded evaluation forms for classmates to score PaperTrail output
vs. plain LLM output on the same topic.
Scoring: 1-5 scale for gap quality, review quality, and overall usefulness.
"""
import json, sys, time, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm import ask_text
import config

EVAL_TOPIC = "transformer models for time-series forecasting"

def generate_plain_llm_output(topic):
    """Generate a plain LLM literature review (no retrieval pipeline)."""
    review_prompt = (
        f"Write a comprehensive academic literature review about '{topic}'. "
        f"Include: an introduction, analysis of key approaches, comparison of methods, "
        f"research gaps, and a conclusion. Cite specific papers using author names. "
        f"The review should be about 500-800 words."
    )
    review = ask_text(review_prompt)

    gaps_prompt = (
        f"List 4-5 specific research gaps in the field of '{topic}'. "
        f"For each gap: describe the gap, explain why it matters, "
        f"and propose a concrete research question. Be specific and cite papers."
    )
    gaps = ask_text(gaps_prompt)

    return {"review": review, "gaps": gaps}

def create_blinded_forms(papertrail_run_path, out_dir="eval/results/human_eval"):
    """Create blinded evaluation forms randomizing which is System A vs B."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load PaperTrail output
    pt_review, pt_gaps = "", ""
    if Path(papertrail_run_path).exists():
        with open(papertrail_run_path, "r", encoding="utf-8") as f:
            pt_run = json.load(f)
        pt_review = pt_run.get("review", "")
        pt_gaps = "\n".join(
            f"- [{g['type']}] {g['finding']}: {g.get('explanation', '')} "
            f"RQ: {g.get('research_question', '')}"
            for g in pt_run.get("gaps", [])
        )

    # Generate plain LLM output
    print("Generating plain LLM output for comparison...")
    plain = generate_plain_llm_output(EVAL_TOPIC)

    # Randomize assignment
    systems = [
        {"id": "papertrail", "review": pt_review, "gaps": pt_gaps},
        {"id": "plain_llm", "review": plain["review"], "gaps": plain["gaps"]}
    ]
    random.shuffle(systems)
    label_map = {systems[0]["id"]: "System A", systems[1]["id"]: "System B"}

    # Save the key (not shown to evaluators)
    key = {"System A": systems[0]["id"], "System B": systems[1]["id"]}
    with open(f"{out_dir}/answer_key.json", "w") as f:
        json.dump(key, f, indent=2)

    # Create the evaluation form
    form = {
        "instructions": (
            "You are evaluating two systems that generated a literature review "
            f"about '{EVAL_TOPIC}'. Rate each system on a scale of 1-5 "
            "(1=very poor, 2=poor, 3=acceptable, 4=good, 5=excellent). "
            "Do NOT look at the answer key file."
        ),
        "topic": EVAL_TOPIC,
        "evaluator_name": "TODO: your name",
        "systems": []
    }

    for sys_info in systems:
        label = label_map[sys_info["id"]]
        form["systems"].append({
            "label": label,
            "review": sys_info["review"],
            "gaps": sys_info["gaps"],
            "scores": {
                "review_coverage": {"score": None, "description": "How well does the review cover the key works and themes? (1-5)"},
                "review_accuracy": {"score": None, "description": "Are the claims accurate and well-supported by citations? (1-5)"},
                "review_coherence": {"score": None, "description": "Is the review well-organized and coherent? (1-5)"},
                "gap_quality": {"score": None, "description": "Are the identified gaps genuine and non-trivial? (1-5)"},
                "gap_actionability": {"score": None, "description": "Do the gaps suggest concrete, actionable research questions? (1-5)"},
                "overall": {"score": None, "description": "Overall usefulness as a research starting point (1-5)"}
            },
            "free_comments": ""
        })

    with open(f"{out_dir}/eval_form.json", "w", encoding="utf-8") as f:
        json.dump(form, f, indent=2)

    # Create 3 copies for 3 evaluators
    for i in range(1, 4):
        evaluator_form = json.loads(json.dumps(form))
        evaluator_form["evaluator_name"] = f"Evaluator {i}"
        with open(f"{out_dir}/eval_form_evaluator_{i}.json", "w", encoding="utf-8") as f:
            json.dump(evaluator_form, f, indent=2)

    print(f"\n{'='*60}")
    print("Human Evaluation Forms Created:")
    print(f"  Answer key: {out_dir}/answer_key.json")
    print(f"  Evaluation form: {out_dir}/eval_form.json")
    print(f"  Evaluator copies: eval_form_evaluator_1/2/3.json")
    print(f"  Label mapping: System A = {key['System A']}, System B = {key['System B']}")

    return key

def aggregate_human_scores(eval_dir="eval/results/human_eval", out_path="eval/results/human_eval_results.json"):
    """Aggregate completed evaluator forms into summary statistics."""
    eval_dir = Path(eval_dir)
    key_path = eval_dir / "answer_key.json"

    if not key_path.exists():
        print("No answer key found. Run create_blinded_forms first.")
        return None

    with open(key_path) as f:
        key = json.load(f)

    all_scores = {"papertrail": {}, "plain_llm": {}}
    evaluator_count = 0

    for form_path in eval_dir.glob("eval_form_evaluator_*.json"):
        with open(form_path) as f:
            form = json.load(f)

        has_scores = False
        for sys_entry in form["systems"]:
            system_id = key[sys_entry["label"]]
            for metric, val in sys_entry["scores"].items():
                score = val.get("score")
                if score is not None:
                    has_scores = True
                    all_scores[system_id].setdefault(metric, []).append(score)

        if has_scores:
            evaluator_count += 1

    if evaluator_count == 0:
        print("No completed evaluator forms found.")
        # Generate simulated scores for the template
        summary = {
            "evaluators": 0,
            "note": "No human evaluations completed yet. Distribute eval_form_evaluator_*.json to 2-3 classmates.",
            "papertrail_scores": {},
            "plain_llm_scores": {},
            "answer_key": key
        }
    else:
        summary = {
            "evaluators": evaluator_count,
            "papertrail_scores": {
                metric: round(sum(scores) / len(scores), 2)
                for metric, scores in all_scores["papertrail"].items()
            },
            "plain_llm_scores": {
                metric: round(sum(scores) / len(scores), 2)
                for metric, scores in all_scores["plain_llm"].items()
            },
            "answer_key": key
        }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Aggregated results saved to: {out_path}")
    return summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Generate blinded forms")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate completed forms")
    parser.add_argument("--run-path", default="eval/provider_comparison/groq_run.json")
    args = parser.parse_args()

    if args.generate:
        create_blinded_forms(args.run_path)
    elif args.aggregate:
        aggregate_human_scores()
    else:
        print("Usage: python eval_human.py --generate  OR  --aggregate")
