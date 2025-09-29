 HotpotQA Dataset

  Official Sources:
  - Primary: https://hotpotqa.github.io/ (official website)
  - GitHub: https://github.com/hotpotqa/hotpot (full repository with scripts)
  - Hugging Face: https://huggingface.co/datasets/hotpotqa/hotpot_qa

  What you get:
  - 113k Wikipedia-based question-answer pairs
  - Multi-hop reasoning questions
  - Supporting facts at sentence level
  - CC BY-SA 4.0 License

  Wikipedia Dataset

  For RAG Research:
  - Hugging Face Mini: https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia
    - 4,118 rows, 852 kB
    - Pre-processed for RAG tasks
  - Full Wikipedia: https://huggingface.co/datasets/wikipedia
    - Complete Wikipedia dump (20220301.en)
    - Much larger but comprehensive

  Commands to download:

  # For HotpotQA
  git clone https://github.com/hotpotqa/hotpot.git
  cd hotpot
  # Follow their download script

  # Or use Python with datasets library:
  pip install datasets

  from datasets import load_dataset

  # HotpotQA
  hotpot = load_dataset("hotpotqa/hotpot_qa", "distractor")

  # Wikipedia RAG mini
  wiki_rag = load_dataset("rag-datasets/rag-mini-wikipedia")

  # Full Wikipedia
  wikipedia = load_dataset("wikipedia", "20220301.en")
  