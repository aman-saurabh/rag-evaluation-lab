from app.graph import ask
from app.retrieve import store

for question in ["What are the four functions of the NIST AI RMF?",
                 "What was Acme's budget for Q4?"]:
    for mode in ["dense", "sparse", "hybrid"]:
        result = ask(question, mode)
        sources = [(source["file"], source["page"]) for source in result["sources"]]
        print(question)
        print(mode, "|", result["answer"][:200], "|", sources)

# Memory: the second question only makes sense with the first ("its" means the GOVERN function).
print("\n-- follow-up in one session (memory ON) --")
for question in ["What is the GOVERN function in the NIST AI RMF?",
                 "What is its purpose?",
                 "And how does the MAP function differ from it?"]:
    result = ask(question, "hybrid", session_id="demo")
    sources = [(source["file"], source["page"]) for source in result["sources"]]
    print(question)
    print("  searched for:", result["search_query"])
    print("  relevant text found:", result["relevant"])
    print("  hybrid", "|", result["answer"][:200], "|", sources)

# Control: the same follow-up without a session. With no memory it cannot know what "its" means.
print("\n-- same follow-up, no session (memory OFF) --")
result = ask("What is its purpose?", "hybrid")
print("  searched for:", result["search_query"])
print("  relevant text found:", result["relevant"])
print("  hybrid", "|", result["answer"][:200])

# Release the Qdrant folder so Windows does not print an error when the script exits.
store.client.close()
