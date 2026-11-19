import pytest
from agent import run_grad_alignment

@pytest.mark.asyncio
async def test_grad_alignment_golden():
    result = await run_grad_alignment("Hans", "Great peer feedback", "100% metrics")
    assert "Section" in result
