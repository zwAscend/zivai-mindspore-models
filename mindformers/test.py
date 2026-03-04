import mindspore as ms
from mindspore import context

try:
    context.set_context(device_target="GPU")
except Exception:
    context.set_context(device_target="CPU")

from mindformers.pipeline import pipeline

gen = pipeline("text_generation", model="mindspore/chatglm2_6b")

prompt = "Explain what knowledge tracing is in one paragraph."
out = gen(prompt, max_length=128, do_sample=True, top_k=40, top_p=0.9)
print(out)
