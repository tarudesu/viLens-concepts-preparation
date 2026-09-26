from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from opencc import OpenCC


MODULE_PATH = Path(__file__).parents[1] / "data" / "build" / "07_backtranslate.py"
spec = importlib.util.spec_from_file_location("step07_backtranslate", MODULE_PATH)
assert spec and spec.loader
step07 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step07)
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "backtranslate_cases.json").read_text(encoding="utf-8"))


def test_norm_french_articles_and_chinese_simplification() -> None:
    converter = OpenCC("t2s")
    assert step07.norm("Le chat.", language="fr", pos="noun", opencc=converter) == "chat"
    assert step07.norm("L'orange.", language="fr", pos="noun", opencc=converter) == "orange"
    assert step07.norm("電腦的。", language="zh", pos="adj", opencc=converter) == "电脑"


def test_backtranslation_pass_canonical_alternative_or_none() -> None:
    converter = OpenCC("t2s")
    assert step07.pass_rule("mèo", ["miêu"], ["mèo."], language="vi", pos="noun", opencc=converter) == (True, "canonical")
    assert step07.pass_rule("mèo", ["miêu"], ["miêu!"], language="vi", pos="noun", opencc=converter) == (True, "alt")
    assert step07.pass_rule("mèo", ["miêu"], ["cat"], language="vi", pos="noun", opencc=converter) == (False, "none")


def test_vietnamese_normalization_and_syllable_containment() -> None:
    assert step07.norm_vi("- HỐ,   sâu!") == "hố sâu"
    assert step07.contains("hố", "cái hố", max_extra_syllables=4)
    assert not step07.contains("ao", "một cái ao ở cuối con đường xa", max_extra_syllables=4)
    assert not step07.contains("hố sâu", "sâu cái hố", max_extra_syllables=4)


def test_second_hop_cleaning_articles_and_verb_marker() -> None:
    assert step07.clean_second_hop("The house!", pos="noun") == "house"
    assert step07.clean_second_hop("to run.", pos="verb") == "run"
    assert step07.clean_second_hop("a tree...", pos="noun") == "tree"


def test_part_b_selection_uses_earliest_beam_then_provisional_order() -> None:
    converter = OpenCC("t2s")
    assert step07.select_part_b("chat", ["félin"], ["félin", "chat", "x", "y", "z"], language="fr", pos="noun", opencc=converter) == ("félin", True, True, ["chat"])
    # Both Chinese forms normalize to the same simplified form, so beam-rank ties retain provisional order.
    assert step07.select_part_b("電腦", ["电脑"], ["电脑", "電腦", "a", "b", "c"], language="zh", pos="noun", opencc=converter) == ("電腦", True, False, ["电脑"])


class MockTranslator:
    """Fixture-only translator that records calls and returns predetermined beams."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def translate_many(self, requests: list[tuple[str, str, str]]) -> dict[tuple[str, str, str], list[str]]:
        output = {}
        for request in requests:
            self.calls.append(request)
            source, target, text = request
            if (source, target, text) == ("vie_Latn", "eng_Latn", "mèo"):
                beam = ["cat"] * 5
            elif (source, target, text) == ("eng_Latn", "vie_Latn", "cat"):
                beam = ["miêu", "khác", "khác", "khác", "khác"]
            elif (source, target) == ("eng_Latn", "fra_Latn"):
                beam = ["félin", "chat", "x", "y", "z"]
            elif (source, target) == ("eng_Latn", "zho_Hans"):
                beam = ["猫", "a", "b", "c", "d"]
            else:
                beam = ["kucing", "a", "b", "c", "d"]
            output[request] = beam
        return output


def test_process_rows_with_mocked_model_covers_parts_a_and_b() -> None:
    row = FIXTURE["row"]
    config = {"lang_codes": {"vi": "vie_Latn", "en": "eng_Latn", "zh": "zho_Hans", "fr": "fra_Latn", "id": "ind_Latn"}, "bt": {"max_extra_syllables": 4}}
    translator = MockTranslator()
    rows, _, info = step07.process_rows([row], translator=translator, config=config, logger=None)
    result = rows[0]
    assert result["bt_pass"] is True
    assert result["bt_pass_strict_v1"] is True
    assert result["rt_contain"] is True
    assert result["fwd_hit"] is True
    assert result["bt_route"] == "strict"
    assert result["bt_match"] == "alt"
    assert result["bt_pass_top1"] is False
    assert result["en_hit"] is True
    assert result["fr_canonical"] == "félin"
    assert result["fr_canonical_changed"] is True
    assert info["pass_by_id"] == {"toy": True}
