from magnum_opus import MagnumOpusEngine, load_model, extract_vectors
model, tok, device = load_model("gpt2")
vectors = extract_vectors(model, tok, device=device)
engine = MagnumOpusEngine(model, tok, vectors, device=device)
print(engine.converse("Hello"))
