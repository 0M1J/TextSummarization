# TextSummarization

## ALGOs - Extractive 

Extractive summarization involves selecting the most important sentences or phrases from the original text and arranging them in a coherent manner to form a summary. Unlike abstractive summarization, which generates new sentences to encapsulate the main ideas, extractive summarization relies on directly extracting existing sentences from the text. This approach typically involves algorithms that rank sentences based on various criteria such as importance, relevance, or similarity to the overall content of the document.

1. **TextRank**:
   - TextRank is an extractive summarization algorithm based on PageRank, which is used by Google to rank web pages. 
   - It works by treating sentences as nodes in a graph, where edges represent the relationship between sentences (such as similarity). 
   - The algorithm iteratively scores sentences based on their importance, similar to how PageRank scores web pages.
   - Sentences with higher scores are selected to form the summary.

2. **LexRank**:
   - LexRank is another graph-based algorithm for extractive summarization.
   - Similar to TextRank, it represents sentences as nodes in a graph, with edges representing similarity based on cosine similarity of their vector representations.
   - However, LexRank also considers the concept of eigenvector centrality, which helps to identify the most important sentences.
   - It ranks sentences based on their centrality scores and selects the top-ranked sentences for the summary.

3. **LSA (Latent Semantic Analysis)**:
   - LSA is a mathematical technique for extracting and representing the contextual meaning of words and documents.
   - It applies singular value decomposition (SVD) to a matrix representation of the text corpus to identify latent semantic structures.
   - In the context of text summarization, LSA identifies the underlying structure of the document and selects sentences that capture the most important semantic information.
   - It does this by reducing the dimensionality of the document-term matrix and retaining the most significant components.

4. **LUHN Algorithm**:
   - The LUHN algorithm is one of the earliest approaches to automated text summarization.
   - It works by identifying and ranking sentences based on their frequency of occurrence of important words.
   - The algorithm assumes that important information is often expressed by repeating certain keywords or phrases.
   - It assigns scores to sentences based on the frequency of occurrence of significant words, and then selects sentences with the highest scores for the summary.
   - While simplistic compared to more advanced algorithms like TextRank and LexRank, LUHN can still provide reasonable summaries for certain types of documents, especially those with clear keyword repetition patterns.
