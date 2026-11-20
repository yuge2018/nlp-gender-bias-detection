"""P5 challenge set v1: source of truth for the 100-item Chinese challenge.

Labels are binary (0 non-biased / 1 biased) under the frozen task definition:
a sentence is biased only when it expresses a stereotype, a role restriction or
derogatory/differential treatment toward a gender group. The mere presence of a
gender word, a neutral description, a denial of a stereotype, or a quotation that
is rebutted in the same sentence is NOT bias.

``targets`` and ``phenomena`` are analysis dimensions and never model features.
This set is a single-annotator draft: ``reviewers`` is empty on purpose until an
independent second annotator reviews every row.
"""
import json
import sys
import time
from pathlib import Path

from src.common.contracts import VERSION, sha256, write_json

DATASET_ID = "p5-gender-challenge-v1"
ANNOTATION_VERSION = "v1.0-draft"
REVIEW_STATUS = "single_annotator_draft"

TARGETS = ("male", "female", "other_explicit", "unspecified")

PHENOMENA = {
    "explicit_stereotype": "显性刻板印象：直接断言某性别的能力、性格或天性",
    "implicit_stereotype": "隐性刻板印象：反问、预设、玩笑或暗示，未直接断言",
    "role_constraint": "角色限制：规定某性别应该/不应该做什么",
    "derogation": "贬损：侮辱、嘲讽或以性别为由贬低",
    "benevolent_sexism": "善意性别歧视：以保护、礼让、赞美的形式固化从属地位",
    "appearance_inference": "外貌推断：由外表推断能力、可靠性或性别气质",
    "victim_blaming": "受害者归责：把侵害归因于当事人的穿着或性别",
    "labor_market_penalty": "职场差别对待：招聘、晋升、岗位分配上的性别门槛",
    "other_identity_derogation": "其他性别身份贬损：对跨性别、非二元等身份的污名化",
    "neutral_description": "中性描述：提到性别但只是陈述事实或身份",
    "denial_of_bias": "否定偏见说法：明确拒绝某一性别刻板要求",
    "quoted_rebuttal": "引用并反驳：句中出现偏见说法，但随即被否定",
    "bias_awareness": "偏见现象讨论：客观指出性别不平等或偏见的危害",
}

_RATIONALE = "按任务定义：表达刻板印象、角色限制或差别对待记1；仅出现性别词、中性描述、否定或引用后反驳记0。"

# (text, label, targets, phenomena)
# 男性对象 40 条（25 正 / 15 负），女性对象 40 条（25 正 / 15 负），
# 其他性别身份 10 条（6 正 / 4 负），多目标 5 条（正），对象不明确 5 条（负）。
ITEMS = [
    # ---------- male: biased ----------
    ("男人哭什么哭，一点样子都没有。", 1, ["male"], ["derogation"]),
    ("男人就该在外面闯，家里的琐事不该沾手。", 1, ["male"], ["role_constraint"]),
    ("真正的男人从不示弱，示弱就是没本事。", 1, ["male"], ["explicit_stereotype"]),
    ("男的天生不适合带孩子，手太糙。", 1, ["male"], ["explicit_stereotype"]),
    ("男人嘛，看见漂亮姑娘走不动路很正常。", 1, ["male"], ["implicit_stereotype"]),
    ("他说话轻声细语，一点男子气概都没有。", 1, ["male"], ["implicit_stereotype", "derogation"]),
    ("男人过了三十还不结婚，多少有点问题。", 1, ["male"], ["implicit_stereotype"]),
    ("男人做家务？那不就是给老婆添乱。", 1, ["male"], ["implicit_stereotype"]),
    ("男人一有钱就变坏，这是常识。", 1, ["male"], ["explicit_stereotype"]),
    ("男孩子不许哭，男子汉要坚强。", 1, ["male"], ["role_constraint"]),
    ("男的当护士，图什么？", 1, ["male"], ["implicit_stereotype", "labor_market_penalty"]),
    ("老公在家带娃做饭，说出去多没面子。", 1, ["male"], ["implicit_stereotype"]),
    ("男人不懂细腻的情感，指望他安慰你是妄想。", 1, ["male"], ["explicit_stereotype"]),
    ("这个男的穿得花花绿绿，娘里娘气的。", 1, ["male"], ["derogation"]),
    ("男人天性花心，是管不住的。", 1, ["male"], ["explicit_stereotype"]),
    ("彩礼都拿不出，还算什么男人。", 1, ["male"], ["derogation", "role_constraint"]),
    ("男的当幼儿园老师，家长能放心吗？", 1, ["male"], ["implicit_stereotype", "labor_market_penalty"]),
    ("他长得文文弱弱，做事肯定靠不住。", 1, ["male"], ["appearance_inference"]),
    ("男人就得让着女人，这叫风度。", 1, ["male"], ["benevolent_sexism"]),
    ("好男不跟女斗，赢了也没面子。", 1, ["male"], ["benevolent_sexism"]),
    ("大老爷们儿还计较这点小事，不像话。", 1, ["male"], ["role_constraint"]),
    ("男人爱打扮就是不务正业。", 1, ["male"], ["explicit_stereotype"]),
    ("男生学文科没出息，还是学理工。", 1, ["male"], ["role_constraint"]),
    ("他哭成那样，太不像个男人了。", 1, ["male"], ["derogation"]),
    ("男人在职场就得狠一点，讲情面成不了事。", 1, ["male"], ["role_constraint"]),
    # ---------- male: non-biased ----------
    ("他是一名幼儿园老师，工作很认真。", 0, ["male"], ["neutral_description"]),
    ("男性也会遭遇家暴，需要社会支持。", 0, ["male"], ["bias_awareness"]),
    ("有人说男人不能哭，我不认同这种说法。", 0, ["male"], ["quoted_rebuttal", "denial_of_bias"]),
    ("她丈夫辞职照顾家庭，是两人商量后的决定。", 0, ["male"], ["neutral_description"]),
    ("这位男护士技术过硬，病人很信任他。", 0, ["male"], ["neutral_description"]),
    ("男性产假应当落实，让父亲真正参与育儿。", 0, ["male"], ["bias_awareness"]),
    ("他喜欢化妆，这是他的个人选择。", 0, ["male"], ["neutral_description"]),
    ("男人之间的友谊也可以很细腻。", 0, ["male"], ["denial_of_bias"]),
    ("他因为被说“不像男人”而难过，这种评价不公平。", 0, ["male"], ["quoted_rebuttal"]),
    ("男生也可以学护理，并且能做得很好。", 0, ["male"], ["denial_of_bias"]),
    ("丈夫承担家务并不丢人。", 0, ["male"], ["denial_of_bias"]),
    ("他拒绝“男人必须养家”的说法，认为家庭责任该共同承担。", 0, ["male"], ["quoted_rebuttal"]),
    ("这位父亲每天接送孩子，邻居早就习以为常。", 0, ["male"], ["neutral_description"]),
    ("男性遭遇性骚扰同样值得重视。", 0, ["male"], ["bias_awareness"]),
    ("他说话温和，同事觉得和他沟通很顺畅。", 0, ["male"], ["neutral_description"]),
    # ---------- female: biased ----------
    ("女人读那么多书有什么用，最后还不是要嫁人。", 1, ["female"], ["explicit_stereotype"]),
    ("女的开车就是不靠谱。", 1, ["female"], ["explicit_stereotype"]),
    ("女人当领导太强势，男同事受不了。", 1, ["female"], ["implicit_stereotype"]),
    ("她升职那么快，背后肯定有人。", 1, ["female"], ["implicit_stereotype", "labor_market_penalty"]),
    ("女人天生不适合学数学。", 1, ["female"], ["explicit_stereotype"]),
    ("女孩子要文静，别整天抛头露面。", 1, ["female"], ["role_constraint"]),
    ("她都三十了还不结婚，人生不完整。", 1, ["female"], ["implicit_stereotype"]),
    ("女的就该顾家，事业心太强不像话。", 1, ["female"], ["role_constraint"]),
    ("女司机一上路就是灾难。", 1, ["female"], ["explicit_stereotype"]),
    ("女人之间没有真友谊，都是塑料姐妹。", 1, ["female"], ["explicit_stereotype"]),
    ("她穿这么少，出了事怪谁？", 1, ["female"], ["victim_blaming"]),
    ("女人情绪化，不能委以重任。", 1, ["female"], ["explicit_stereotype", "labor_market_penalty"]),
    ("当妈的不亲自带孩子，太自私了。", 1, ["female"], ["role_constraint"]),
    ("女博士谁敢娶？", 1, ["female"], ["implicit_stereotype", "derogation"]),
    ("她技术不行吧，毕竟是个女的。", 1, ["female"], ["implicit_stereotype", "labor_market_penalty"]),
    ("女人谈生意，靠的不就是撒娇。", 1, ["female"], ["implicit_stereotype", "derogation"]),
    ("女孩子学理工科，多半跟不上。", 1, ["female"], ["implicit_stereotype"]),
    ("她都四十了还想创业，不如安稳待着。", 1, ["female"], ["role_constraint"]),
    ("女员工一结婚就要生孩子，招她不合算。", 1, ["female"], ["labor_market_penalty"]),
    ("长得好看的女的升职，大家都懂的。", 1, ["female"], ["implicit_stereotype"]),
    ("女人要温柔贤惠，太能干了男人不喜欢。", 1, ["female"], ["benevolent_sexism", "role_constraint"]),
    ("女人吵架就是无理取闹。", 1, ["female"], ["explicit_stereotype"]),
    ("她都当妈了还打扮成这样，像什么话。", 1, ["female"], ["role_constraint"]),
    ("女强人的家庭一定不幸福。", 1, ["female"], ["explicit_stereotype"]),
    ("她讲话那么冲，一点都不像个女的。", 1, ["female"], ["derogation"]),
    # ---------- female: non-biased ----------
    ("她是一名卡车司机，跑长途十年了。", 0, ["female"], ["neutral_description"]),
    ("女性在科研领域的贡献不该被忽视。", 0, ["female"], ["bias_awareness"]),
    ("有人说女生学不好数学，这是没有依据的偏见。", 0, ["female"], ["quoted_rebuttal"]),
    ("她选择不婚，这是她自己的决定。", 0, ["female"], ["neutral_description"]),
    ("这位女工程师负责整个项目。", 0, ["female"], ["neutral_description"]),
    ("她脾气直，但工作上讲道理。", 0, ["female"], ["neutral_description"]),
    ("女性的平均薪酬低于男性，这个差距值得关注。", 0, ["female"], ["bias_awareness"]),
    ("她生完孩子后回到职场，团队给了很多支持。", 0, ["female"], ["neutral_description"]),
    ("别拿“贤惠”要求所有女人。", 0, ["female"], ["denial_of_bias"]),
    ("她喜欢短发和运动装，这很正常。", 0, ["female"], ["neutral_description"]),
    ("女领导也可以果断而公正。", 0, ["female"], ["denial_of_bias"]),
    ("他说“女人不适合当领导”，我当场反驳了。", 0, ["female"], ["quoted_rebuttal"]),
    ("她是两个孩子的妈妈，也是公司的合伙人。", 0, ["female"], ["neutral_description"]),
    ("女性之间的友谊可以很深厚。", 0, ["female"], ["denial_of_bias"]),
    ("她拒绝被说“不像女人”，认为那是刻板印象。", 0, ["female"], ["quoted_rebuttal"]),
    # ---------- other explicit ----------
    ("不男不女的，看着就让人不舒服。", 1, ["other_explicit"], ["other_identity_derogation"]),
    ("跨性别者心理都有问题，最好别打交道。", 1, ["other_explicit"], ["other_identity_derogation"]),
    ("打扮得男女不分，成何体统。", 1, ["other_explicit"], ["other_identity_derogation"]),
    ("ta 既不是男也不是女，想法肯定不正常。", 1, ["other_explicit"], ["other_identity_derogation"]),
    ("跨性别员工要求别人用对名字，就是矫情。", 1, ["other_explicit"], ["other_identity_derogation"]),
    ("非二元性别只是跟风赶时髦，过两年就消停了。", 1, ["other_explicit"], ["other_identity_derogation", "implicit_stereotype"]),
    ("我同事是跨性别男性，工作能力很强。", 0, ["other_explicit"], ["neutral_description"]),
    ("ta 用“他们”自称，我尊重 ta 的选择。", 0, ["other_explicit"], ["neutral_description"]),
    ("公司为跨性别员工修改了更衣室使用规则。", 0, ["other_explicit"], ["neutral_description"]),
    ("不管男女还是非二元，能力才是标准。", 0, ["other_explicit"], ["denial_of_bias"]),
    # ---------- multi-target (biased) ----------
    ("男人理性，女人感性，这是天性。", 1, ["male", "female"], ["explicit_stereotype"]),
    ("男主外女主内，天经地义。", 1, ["male", "female"], ["role_constraint"]),
    ("男女搭配干活不累，女的负责接待就行。", 1, ["male", "female"], ["role_constraint"]),
    ("招聘时男的看能力，女的看形象。", 1, ["male", "female"], ["labor_market_penalty"]),
    ("男生就该让着女生。", 1, ["male", "female"], ["benevolent_sexism"]),
    # ---------- unspecified (non-biased) ----------
    ("这个社会对男女的要求不一样，但不该如此。", 0, ["unspecified"], ["bias_awareness"]),
    ("男性和女性的差异常被过度强调。", 0, ["unspecified"], ["bias_awareness"]),
    ("不管什么性别，家务都该一起分担。", 0, ["unspecified"], ["denial_of_bias"]),
    ("性别不该决定一个人的职业选择。", 0, ["unspecified"], ["denial_of_bias"]),
    ("我们组有男有女，分工按专长来。", 0, ["unspecified"], ["neutral_description"]),
]


def make_rows():
    """Return (challenge, labels, annotations) rows with stable ids."""
    challenge, labels, annotations = [], [], []
    for index, (text, label, targets, phenomena) in enumerate(ITEMS, 1):
        ident = f"p5-{index:03d}"
        if not text.strip():
            raise ValueError(f"{ident}: empty text")
        if label not in (0, 1):
            raise ValueError(f"{ident}: label must be 0/1")
        if not targets or any(target not in TARGETS for target in targets):
            raise ValueError(f"{ident}: unknown target in {targets}")
        if not phenomena or any(phenomenon not in PHENOMENA for phenomenon in phenomena):
            raise ValueError(f"{ident}: unknown phenomenon in {phenomena}")
        challenge.append({"id": ident, "text": text})
        labels.append({"id": ident, "label": label})
        annotations.append({
            "id": ident,
            "targets": list(targets),
            "phenomena": list(phenomena),
            "reviewers": [],
            "adjudication": f"draft-v1 单标注者（AI 辅助起草），双人独立复核与分歧裁决待完成。{_RATIONALE}",
        })
    return challenge, labels, annotations


def composition(rows):
    """Counts by target x label and phenomenon x label; used by report and tests."""
    challenge, labels, annotations = rows
    by_target = {target: {"0": 0, "1": 0} for target in TARGETS}
    by_phenomenon = {key: {"0": 0, "1": 0} for key in PHENOMENA}
    label_of = {row["id"]: row["label"] for row in labels}
    for annotation in annotations:
        label = str(label_of[annotation["id"]])
        for target in annotation["targets"]:
            by_target[target][label] += 1
        for phenomenon in annotation["phenomena"]:
            by_phenomenon[phenomenon][label] += 1
    return {
        "n": len(challenge),
        "positive": sum(row["label"] for row in labels),
        "negative": sum(1 - row["label"] for row in labels),
        "by_target": by_target,
        "by_phenomenon": by_phenomenon,
        "multi_target": sum(1 for row in annotations if len(row["targets"]) > 1),
    }


def build_challenge(output_dir):
    """Write challenge/labels/annotations and a sealed manifest; refuse to overwrite mismatched content."""
    out = Path(output_dir)
    rows = make_rows()
    files = {"challenge.jsonl": rows[0], "challenge_labels.jsonl": rows[1], "challenge_annotations.jsonl": rows[2]}
    payloads = {}
    for name, records in files.items():
        body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
        payloads[name] = body
    out.mkdir(parents=True, exist_ok=True)
    for name, body in payloads.items():
        target = out / name
        if target.exists() and target.read_text(encoding="utf-8") != body:
            raise ValueError(f"{target}: differs from generator output; choose a new output directory or review the change")
        target.write_text(body, encoding="utf-8", newline="\n")
    manifest = {
        "schema_version": VERSION,
        "dataset_id": DATASET_ID,
        "is_mock": False,
        "annotation_version": ANNOTATION_VERSION,
        "review_status": REVIEW_STATUS,
        "label_map": {"0": "non_biased", "1": "biased"},
        "targets": list(TARGETS),
        "phenomena": PHENOMENA,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S+08:00", time.localtime()),
        "freeze_note": "冻结于模型查看之前；标签与对象/现象分开保存，训练接口不加载本目录。",
        "files": {
            name: {"sha256": sha256(out / name), "rows": len(records)}
            for name, records in files.items()
        },
        "composition": composition(rows),
    }
    write_json(out / "manifest.json", manifest)
    return {"output_dir": str(out), "rows": len(rows[0]), "manifest_sha256": sha256(out / "manifest.json"), "composition": manifest["composition"]}


def seal_challenge(challenge_dir):
    """Recompute file hashes and freeze time after an explicit annotation revision."""
    base = Path(challenge_dir)
    manifest_path = base / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = {}
    for name in ("challenge.jsonl", "challenge_labels.jsonl", "challenge_annotations.jsonl"):
        path = base / name
        if not path.is_file():
            raise ValueError(f"{path}: missing annotation file")
        with path.open(encoding="utf-8") as stream:
            rows = sum(1 for _ in stream)
        files[name] = {"sha256": sha256(path), "rows": rows}
    manifest["files"] = files
    manifest["frozen_at"] = time.strftime("%Y-%m-%dT%H:%M:%S+08:00", time.localtime())
    write_json(manifest_path, manifest)
    return {"challenge_dir": str(base), "files": files, "frozen_at": manifest["frozen_at"]}


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Build or reseal the P5 challenge set")
    parser.add_argument("command", choices=("build", "seal"))
    parser.add_argument("--output-dir", default="data/challenge/" + DATASET_ID)
    parser.add_argument("--challenge-dir", default=None)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_challenge(args.output_dir)
        else:
            result = seal_challenge(args.challenge_dir or args.output_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(f"P5 challenge error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
