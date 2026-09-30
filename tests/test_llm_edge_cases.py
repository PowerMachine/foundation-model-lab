import torch

from fmlab.llm.data import ByteTokenizer, InstructionDataset, InstructionExample


def test_sft_long_prompt_keeps_supervised_response_token() -> None:
    dataset = InstructionDataset(
        [InstructionExample(prompt="아주 긴 질문" * 100, response="정답")],
        ByteTokenizer(),
        max_length=16,
    )
    labels = dataset[0]["labels"]
    assert torch.any(labels.ne(-100))
    assert labels[labels.ne(-100)][-1].item() == ByteTokenizer.eos_token_id
