"""Official checkpoint layouts and target cache contracts, without downloads."""

from copy import deepcopy
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.generate import generate_step
from mlx_vlm.models.base import InputEmbeddingsFeatures
from mlx_vlm.models.cache import make_prompt_cache
from mlx_vlm.speculative.common import verify_forward
from mlx_vlm.speculative.drafters import load_drafter
from mlx_vlm.speculative.drafters.compatibility import validate_dflash_target
from mlx_vlm.speculative.drafters.dspark import Model as DSpark
from mlx_vlm.speculative.drafters.dspark import ModelConfig as DSparkConfig
from mlx_vlm.speculative.drafters.mtp_split import get_mtp_splitter
from mlx_vlm.tests.test_models import model_case
from mlx_vlm.tests.test_speculative import equal, module, split_checkpoint
from mlx_vlm.utils import get_model_type


def dspark_config(layout="speculators", **overrides):
    backbone = dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=128,
    )
    raw = dict(block_size=4, mask_token_id=63, markov_rank=8)
    if layout == "speculators":
        raw.update(
            speculators_model_type="dspark",
            transformer_layer_config=backbone,
            aux_hidden_state_layer_ids=[0, 1],
        )
    else:
        raw.update(
            backbone,
            model_type="qwen3",
            architectures=["Qwen3DSparkModel"],
            target_layer_ids=[0, 1],
            num_target_layers=4,
        )
    return {**raw, **overrides}


@pytest.mark.parametrize("layout", ["speculators", "openbmb"])
@pytest.mark.parametrize("anchor", [True, False])
def test_official_dspark_config(layout, anchor):
    raw = dspark_config(layout, sample_from_anchor=anchor)
    assert get_model_type(raw) == "dspark"
    cfg = DSparkConfig.from_dict(raw)
    assert cfg.hidden_size == 32 and cfg.target_layer_ids == [0, 1]
    assert cfg.block_size == (4 if layout == "speculators" and not anchor else 5)
    assert cfg.num_target_layers == (None if layout == "speculators" else 4)
    target = SimpleNamespace(
        config=SimpleNamespace(hidden_size=32, vocab_size=64),
        model=SimpleNamespace(layers=[None] * 4),
        supports_speculative_decoding=True,
    )
    validate_dflash_target(cfg, target, "DSpark")
    target.model.layers = [None]
    with pytest.raises(ValueError, match="layer"):
        validate_dflash_target(cfg, target, "DSpark")


def test_dspark_keeps_checkpoint_output_head():
    cfg = DSparkConfig.from_dict(dspark_config())
    draft = DSpark(cfg)
    weights = dict(tree_flatten(draft.parameters()))
    weights["lm_head.weight"] = mx.random.normal((64, 32))
    draft.load_weights(list(draft.sanitize(weights).items()), strict=True)
    target = SimpleNamespace(
        config=SimpleNamespace(hidden_size=32, vocab_size=64),
        model=SimpleNamespace(layers=[None] * 4, embed_tokens=nn.Embedding(64, 32)),
        lm_head=nn.Linear(32, 64, bias=False),
        supports_speculative_decoding=True,
    )
    draft.bind(target)
    equal(draft.lm_head.weight, weights["lm_head.weight"])
    assert draft.lm_head is not target.lm_head


def test_dspark_reduced_vocabulary_conditions_on_target_ids(monkeypatch):
    draft = DSpark(DSparkConfig.from_dict(dspark_config(draft_vocab_size=4)))
    weights = dict(tree_flatten(draft.parameters()))
    weights.update(
        d2t=mx.array([6, 8, 10, 12], dtype=mx.int64),
        t2d=mx.zeros((64,), dtype=mx.bool_),
    )
    weights["lm_head.weight"] = mx.zeros((4, 32))
    weights["markov_head.markov_w1.weight"] = mx.zeros((64, 8))
    for row, column in zip([63, 6, 9, 12], range(4)):
        weights["markov_head.markov_w1.weight"][row, column] = 1
    weights["markov_head.markov_w2.weight"] = mx.concatenate(
        [mx.eye(4), mx.zeros((4, 4))], axis=1
    )
    draft.load_weights(list(draft.sanitize(weights).items()), strict=True)
    monkeypatch.setattr(
        draft, "_hidden", lambda inputs, *args: mx.zeros((*inputs.shape, 32))
    )
    tokens = draft.draft_block(
        63, None, None, 5, lambda logits: mx.argmax(logits, axis=-1)
    )
    assert tokens.tolist() == [[6, 9, 12, 15]]


@pytest.mark.parametrize("non_causal", [False, True])
def test_speculators_sliding_attention_causality(non_causal):
    raw = dspark_config(sliding_window_non_causal=non_causal)
    raw["transformer_layer_config"].update(
        layer_types=["sliding_attention", "full_attention"], sliding_window=8
    )
    draft = DSpark(DSparkConfig.from_dict(raw))
    x, context = mx.random.normal((1, 3, 32)), mx.random.normal((1, 3, 32))
    changed = mx.concatenate([x[:, :2], x[:, 2:] + 10], axis=1)
    for i, layer in enumerate(draft.layers):

        def forward(value):
            return layer.self_attn(value, context, draft.rope, draft.make_cache()[i])[
                :, 0
            ]

        difference = mx.max(mx.abs(forward(x) - forward(changed))).item()
        assert (difference < 1e-6) == (i == 0 and not non_causal)


def target_model(family):
    basic = dict(
        model_type=family,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        rms_norm_eps=1e-5,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        head_dim=8,
        rope_theta=10000.0,
    )
    if family == "deepseek_v41":
        raw = deepcopy(model_case(family)["config"])
        raw.update(
            dspark_target_layer_ids=[1, 2, 3],
            num_nextn_predict_layers=2,
            dspark_n_routed_experts=2,
            dspark_num_experts_per_tok=1,
            dspark_noise_token_id=63,
            dspark_markov_rank=8,
            dspark_block_size=3,
        )
        cfg = module(f"models.{family}.config").ModelConfig.from_dict(raw)
    elif family == "mistral4":
        raw = model_case("TestModels.mistral4")["config"]["text_config"]
        cfg = module("models.mistral3.config").TextConfig.from_dict(raw)
        family = "mistral3"
    elif family == "kimi_k3":
        basic.update(
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=8,
            attn_res_block_size=2,
            linear_attn_config=dict(
                kda_layers=[1, 3], num_heads=2, head_dim=32, short_conv_kernel_size=4
            ),
        )
        cfg = module(f"models.{family}.config").TextConfig.from_dict(basic)
    elif family == "mellum":
        basic.update(
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            head_dim=8,
            norm_topk_prob=True,
            sliding_window=4,
            layer_types=["sliding_attention"] * 3 + ["full_attention"],
            rope_parameters={
                kind: {"rope_theta": 10000.0}
                for kind in ("sliding_attention", "full_attention")
            },
            num_nextn_predict_layers=1,
        )
        cfg = module(f"models.{family}.config").ModelConfig.from_dict(basic)
    else:
        cfg = module(f"models.{family}.config").ModelConfig.from_dict(basic)
    model = module(f"models.{family}.language").LanguageModel(cfg)
    if family == "deepseek_v41":
        from mlx_vlm.models.deepseek_v41.engram import NgramHashState

        model.engram_hash = NgramHashState(
            cfg, model.layout, token_map=[i % 7 for i in range(cfg.vocab_size)]
        )
    model.eval()
    return model


@pytest.mark.parametrize(
    "family", ["qwen3", "llama", "mistral4", "kimi_k3", "deepseek_v41"]
)
@pytest.mark.parametrize("keep", [0, 1, 3])
def test_target_hidden_capture_and_rejected_prefix(family, keep):
    mx.random.seed(5)
    model = target_model(family)
    ordinary, speculative = make_prompt_cache(model), make_prompt_cache(model)
    prompt, block = mx.array([[1, 2, 3, 4, 5, 6, 7]]), mx.array([[8, 9, 10]])
    taps = [0, 1]
    first = model(prompt, cache=ordinary, capture_layer_ids=taps)
    equal(first.logits, model(prompt, cache=speculative).logits)
    assert len(first.hidden_states) == len(taps)
    output, transaction = verify_forward(
        model, block, speculative, capture_layer_ids=taps
    )
    expected = [
        model(block[:, i : i + 1], cache=ordinary, capture_layer_ids=taps)
        for i in range(keep)
    ]
    mx.eval(output.logits)
    transaction.commit([keep])
    if keep:
        equal(
            output.logits[:, :keep],
            mx.concatenate([out.logits for out in expected], axis=1),
            atol=1e-4,
            rtol=1e-4,
        )
    equal(
        [c.state for c in speculative],
        [c.state for c in ordinary],
        atol=1e-4,
        rtol=1e-4,
    )
    equal(
        model(mx.array([[11]]), cache=speculative).logits,
        model(mx.array([[11]]), cache=ordinary).logits,
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize("abort", [False, True])
def test_deepseek_v41_batch_cache_restores_all_state(abort):
    model = target_model("deepseek_v41")
    prompts = [mx.array([[1, 2, 3, 4, 5, 6, 7]]), mx.array([[2, 3, 4, 5]])]
    caches = [model.make_cache()[0] for _ in prompts]
    expected = [model.make_cache()[0] for _ in prompts]
    for prompt, actual, reference in zip(prompts, caches, expected):
        model(prompt, cache=[actual])
        model(prompt, cache=[reference])
    batch = type(caches[0]).merge(caches)
    block = mx.array([[8, 9, 10], [5, 6, 7]])
    _, transaction = verify_forward(model, block, [batch])
    if abort:
        transaction.abort()
    else:
        transaction.commit([1, 3])
    for i, count in enumerate([0, 0] if abort else [1, 3]):
        for j in range(count):
            model(block[i : i + 1, j : j + 1], cache=[expected[i]])
        equal(batch.rows[i].state, expected[i].state, atol=1e-5, rtol=1e-5)
        assert batch.rows[i].offset == expected[i].offset
    assert not batch.is_speculating and not any(
        row.is_speculating for row in batch.rows
    )


def test_native_hidden_state_contracts():
    ids = mx.array([[1, 2, 3]])
    target = target_model("mellum")
    hidden = target(ids, return_hidden=True).hidden_states[-1]
    equal(hidden, target.model(ids))
    target = target_model("deepseek_v41")
    hidden = target(ids, capture_layer_ids=[0]).hidden_states[0]
    # V4.1 taps precede the block; the first tap is the token embedding.
    equal(hidden, target.embed_tokens(ids))


@pytest.mark.parametrize("family", ["mellum", "deepseek_v41"])
def test_native_split_load_and_greedy_generation(family, tmp_path):
    mx.random.seed(7)
    target = target_model(family)
    raw = target.config.to_dict()
    arch = module(
        f"speculative.drafters.{family}_{'mtp' if family == 'mellum' else 'dspark'}"
    )
    opts = {} if family == "mellum" else dict(n_mtp_layers=2, markov_rank=8)
    draft = arch.Model(arch.ModelConfig(text_config=raw, **opts))
    source = {}
    for key, value in tree_flatten(draft.parameters()):
        if family == "mellum":
            key = f"mtp.{key}"
        elif key.startswith("markov_head."):
            key = f"mtp.1.{key.replace('markov_w1', 'embed').replace('markov_w2', 'head')}"
        else:
            key = key.replace("stages.", "mtp.", 1)
        if ".switch_mlp." in key:
            for index, tensor in enumerate(value):
                stem = key.replace(".switch_mlp.", f".experts.{index}.")
                if family == "deepseek_v41":
                    stem = (
                        stem.replace("gate_proj", "w1")
                        .replace("down_proj", "w2")
                        .replace("up_proj", "w3")
                    )
                source[stem] = tensor
        elif ".attn.wo_a.weight" in key:
            source[key] = value.flatten(0, 1)
        else:
            source[key] = value
    config, weights, path = split_checkpoint(tmp_path, raw, source, separate=True)
    loaded, kind = load_drafter(str(path))
    bundled, bundled_kind = load_drafter(str(tmp_path))
    assert kind == bundled_kind == ("mtp" if family == "mellum" else "dflash")
    equal(
        dict(tree_flatten(loaded.parameters())),
        dict(tree_flatten(bundled.parameters())),
    )
    ids = mx.array([[1, 2, 3, 4, 5, 6, 7]])
    embedding = getattr(target, "model", target).embed_tokens
    outer = SimpleNamespace(
        language_model=target,
        config=target.config,
        get_input_embeddings=lambda ids, *args, **kwargs: InputEmbeddingsFeatures(
            inputs_embeds=embedding(ids)
        ),
    )

    def run(**kwargs):
        return [
            int(token)
            for token, _ in generate_step(
                ids, outer, None, None, max_tokens=8, temperature=0, **kwargs
            )
        ]

    assert run() == run(draft_model=loaded, draft_kind=kind, num_draft_tokens=3)
    assert loaded.draft_lens


@pytest.mark.parametrize("name", ["qwen3_5_text", "qwen3_5_moe_text"])
def test_qwen_text_aliases_share_native_splitter(name):
    splitter = get_mtp_splitter(
        source_config={"model_type": name, "mtp_num_hidden_layers": 1}
    )
    assert splitter.output_model_type == "qwen3_5_mtp"


@pytest.mark.parametrize("bits", [4, 8])
def test_deepseek_v41_native_packed_weights(bits):
    split = get_mtp_splitter("deepseek_v41")
    raw = target_model("deepseek_v41").config.to_dict()
    mode = f"mxfp{bits}"
    original = mx.random.normal((32, 32))
    packed, scales = mx.quantize(original, group_size=32, bits=bits, mode=mode)
    prefix = "mtp.0.main_proj" if bits == 8 else "mtp.0.ffn.experts.0.w1"
    source = {f"{prefix}.weight": packed.view(mx.uint8), f"{prefix}.scale": scales}
    if bits == 4:
        source.update(
            {k.replace("experts.0", "experts.1"): v for k, v in list(source.items())}
        )
    converted = split.transform(source, raw, False)
    name = "stages.0.main_proj" if bits == 8 else "stages.0.ffn.switch_mlp.gate_proj"
    weight, scale = converted[f"{name}.weight"], converted[f"{name}.scales"]
    if bits == 4:
        weight, scale = weight[0], scale[0]
    equal(
        mx.dequantize(weight, scale, group_size=32, bits=bits, mode=mode),
        mx.dequantize(packed, scales, group_size=32, bits=bits, mode=mode),
    )
    assert split.quantization_from_source(converted, {})[name]["mode"] == mode
