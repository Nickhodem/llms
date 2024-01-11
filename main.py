from transformers import pipeline

# transcriber = pipeline(task="automatic-speech-recognition")
# transcript = transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
# print(transcript)
# #
# transcriber = pipeline(model="openai/whisper-large-v2")
# print(transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"))

def data():
    for i in range(100):
        yield f"My example {i}"


pipe = pipeline(model="gpt2", device=0)
generated_characters = 0
for out in pipe(data()):
    print(out[0]["generated_text"])
    generated_characters += len(out[0]["generated_text"])