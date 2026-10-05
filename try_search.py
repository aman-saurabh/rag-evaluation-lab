from app.retrieve import search, store

for question in ["What are the four functions of the AI RMF?",
                 "SP 800-218 PS.1.1",
                 "how do I keep my software from being tampered with"]:
    for mode in ["dense", "sparse", "hybrid"]:
        print("\n", question, "|", mode)
        for result in search(question, mode):
            print(f"  {result['file']} p{result['page']}  {result['text'][:80]!r}")

# Release the Qdrant folder so Windows does not print an error when the script exits.
store.client.close()
