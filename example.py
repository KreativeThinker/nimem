from nimem.core.text_processing import extract_triplets, process_text_pipeline

text = "John works for Apple Inc. and lives in San Francisco. Alice founded SpaceX."

# Default: spaCy NER (en_core_web_md) + heuristic relation extraction
result = extract_triplets(text)
print("=== Heuristic ===")
for t in result.unwrap():
    print(f"  {t.subject} --[{t.relation}]--> {t.object}")

# GLiNER2 native relation extraction
# result = extract_triplets(text, use_gliner2=True)
# print("\n=== GLiNER2 ===")
# for t in result.unwrap():
#     print(f"  {t.subject} --[{t.relation}]--> {t.object}")

# Full pipeline with coreference resolution
long_text = "John works for Apple Inc. He lives in San Francisco."
resolved, triplets = process_text_pipeline(
    long_text, use_coref=False, use_gliner2=False
).unwrap()
print(f"\n=== Pipeline (coref) ===")
print(f"Resolved: {resolved}")
for t in triplets:
    print(f"  {t.subject} --[{t.relation}]--> {t.object}")
