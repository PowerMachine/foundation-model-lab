from pathlib import Path

from fmlab.llm.tokenizer_lab import train_and_compare_tokenizers


def test_bpe_lab_writes_visual_and_round_trips(tmp_path: Path) -> None:
    result = train_and_compare_tokenizers(tmp_path / "bpe", vocab_size=300)
    assert result.metrics["all_round_trips_equal"]
    assert result.metrics["average_bpe_tokens"] < result.metrics["average_byte_tokens"]
    assert (tmp_path / "bpe/token_lengths.png").is_file()
    assert (tmp_path / "bpe/report.html").is_file()
