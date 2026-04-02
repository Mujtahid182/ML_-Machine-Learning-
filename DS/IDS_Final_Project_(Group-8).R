library(tm)
library(readr)
library(SnowballC)
library(stringi)
library(textclean)
library(textstem)
library(hunspell)
library(ggplot2)
library(wordcloud)
library(RColorBrewer)
library(proxy)

set.seed(123)

human_df <- read.csv(
  "E:/Data_Science Datasets/Final Dataset for final term/Human_vs_LLM_Papers.csv",
  header = TRUE
)

generated_df <- read.csv(
  "E:/Data_Science Datasets/Final Dataset for final term/Generated_Dataset.csv",
  header = TRUE
)

bg_text <- human_df$Abstract
fg_text <- generated_df$abstract

bg_text <- bg_text[!is.na(bg_text) & nzchar(bg_text)]
fg_text <- fg_text[!is.na(fg_text) & nzchar(fg_text)]

cat("Background docs (Human):", length(bg_text), "\n")
cat("Foreground docs (LLM):", length(fg_text), "\n")

labels <- c(rep("Human", length(bg_text)), rep("LLM", length(fg_text)))
cat("Label counts:\n"); print(table(labels))


bg_corpus_raw <- VCorpus(VectorSource(bg_text))
fg_corpus_raw <- VCorpus(VectorSource(fg_text))


convert_to_utf8 <- function(corpus) {
  tm_map(corpus, content_transformer(function(x) {
    x <- stringi::stri_enc_toutf8(x)
    iconv(x, from = "UTF-8", to = "UTF-8", sub = " ")
  }))
}

bg_corpus <- convert_to_utf8(bg_corpus_raw)
fg_corpus <- convert_to_utf8(fg_corpus_raw)


clean_corpus <- function(corpus) {
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, content_transformer(textclean::replace_contraction))
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removeWords, stopwords("en"))
  corpus <- tm_map(corpus, stripWhitespace)
  corpus
}

bg_clean <- clean_corpus(bg_corpus)
fg_clean <- clean_corpus(fg_corpus)

cat("\nSAMPLE CHECK\n")
cat("BEFORE CLEANING:\n", substr(bg_corpus[[1]]$content, 1, 250), "\n\n")
cat("AFTER CLEANING:\n", substr(bg_clean[[1]]$content, 1, 250), "\n\n")


tokenize_corpus <- function(corpus) {
  lapply(corpus, function(d) unlist(strsplit(d$content, "\\s+")))
}

bg_tokens <- tokenize_corpus(bg_clean)
fg_tokens <- tokenize_corpus(fg_clean)

bg_tokens_lemma <- lapply(bg_tokens, textstem::lemmatize_words)
fg_tokens_lemma <- lapply(fg_tokens, textstem::lemmatize_words)


spell_check_tokens <- function(tokens_list) {
  lapply(tokens_list, function(toks) {
    bad <- hunspell(toks)
    unique(toks[lengths(bad) > 0])
  })
}

bg_misspelled <- spell_check_tokens(bg_tokens_lemma)
fg_misspelled <- spell_check_tokens(fg_tokens_lemma)

cat("Background doc-1 misspelled (sample):\n")
print(head(bg_misspelled[[1]], 20))
cat("\nForeground doc-1 misspelled (sample):\n")
print(head(fg_misspelled[[1]], 20))


bg_text_lemma <- sapply(bg_tokens_lemma, paste, collapse = " ")
fg_text_lemma <- sapply(fg_tokens_lemma, paste, collapse = " ")

all_text_lemma <- c(bg_text_lemma, fg_text_lemma)
all_corpus_lemma <- VCorpus(VectorSource(all_text_lemma))


dtm_tfidf <- DocumentTermMatrix(
  all_corpus_lemma,
  control = list(
    weighting = weightTfIdf,
    wordLengths = c(3, Inf)
  )
)

cat("\nOriginal DTM dims:\n")
print(dim(dtm_tfidf))


dtm_tfidf <- removeSparseTerms(dtm_tfidf, 0.95)  
cat("\nAfter removeSparseTerms dims:\n")
print(dim(dtm_tfidf))

tfidf_matrix <- as.matrix(dtm_tfidf)


n_bg <- length(bg_text_lemma)
bg_matrix <- tfidf_matrix[1:n_bg, , drop = FALSE]
fg_matrix <- tfidf_matrix[(n_bg + 1):nrow(tfidf_matrix), , drop = FALSE]


bg_avg <- colMeans(bg_matrix)
fg_avg <- colMeans(fg_matrix)

contrast_score <- bg_avg - fg_avg

top_bg_terms <- head(sort(contrast_score, decreasing = TRUE), 30)  
top_fg_terms <- head(sort(contrast_score, decreasing = FALSE), 30) 

cat("\nTop Human-specific terms:\n")
print(top_bg_terms)

cat("\nTop LLM-specific terms:\n")
print(top_fg_terms)


contrast_df <- data.frame(term = names(contrast_score), score = as.numeric(contrast_score))

ggplot(contrast_df[order(-contrast_df$score)[1:30], ],
       aes(x = reorder(term, score), y = score)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  ggtitle("Top 30 Human-Specific Terms") +
  theme_minimal()

ggplot(contrast_df[order(contrast_df$score)[1:30], ],
       aes(x = reorder(term, score), y = score)) +
  geom_bar(stat = "identity", fill = "tomato") +
  coord_flip() +
  ggtitle("Top 30 LLM-Specific Terms") +
  theme_minimal()


par(mfrow = c(1,2))
wordcloud(words = names(top_bg_terms), freq = abs(top_bg_terms),
          max.words = 30, random.order = FALSE, colors = brewer.pal(8, "Blues"))
title("Human-Specific WordCloud")

wordcloud(words = names(top_fg_terms), freq = abs(top_fg_terms),
          max.words = 30, random.order = FALSE, colors = brewer.pal(8, "Reds"))
title("LLM-Specific WordCloud")
par(mfrow = c(1,1))



row_l2_normalize <- function(x){
  s <- sqrt(rowSums(x^2)) + 1e-12
  x / s
}

tfidf_norm <- row_l2_normalize(tfidf_matrix)

k <- 2
kmeans_result <- kmeans(tfidf_norm, centers = k, nstart = 50)

cat("\nK-means cluster table:\n")
print(table(labels, kmeans_result$cluster))


dist_matrix <- proxy::dist(tfidf_norm, method = "cosine")
hc <- hclust(dist_matrix, method = "average")

plot(hc, labels = FALSE, hang = -1, main = "Hierarchical Dendrogram (Cosine + Average)")
hc_clusters <- cutree(hc, k = 2)

cat("\nHierarchical cluster table:\n")
print(table(labels, hc_clusters))


purity <- function(true_labels, clusters){
  tab <- table(true_labels, clusters)
  sum(apply(tab, 2, max)) / length(true_labels)
}

purity_kmeans <- purity(labels, kmeans_result$cluster)
purity_hc     <- purity(labels, hc_clusters)

cat("\nPurity (K-means):", purity_kmeans, "\n")
cat("Purity (Hierarchical):", purity_hc, "\n")


get_top_words <- function(dtm_mat, clusters, top_n = 10){
  cl <- sort(unique(clusters))
  res <- lapply(cl, function(k){
    idx <- which(clusters == k)
    avg <- colMeans(dtm_mat[idx, , drop = FALSE])
    head(sort(avg, decreasing = TRUE), top_n)
  })
  names(res) <- paste0("Cluster_", cl)
  res
}

top_words_kmeans <- get_top_words(tfidf_matrix, kmeans_result$cluster, 10)
top_words_hc     <- get_top_words(tfidf_matrix, hc_clusters, 10)

cat("\nTop 10 words per K-means cluster:\n")
print(top_words_kmeans)

cat("\nTop 10 words per Hierarchical cluster:\n")
print(top_words_hc)


plot_top_words <- function(top_words_list, title_prefix="K-means"){
  for(i in seq_along(top_words_list)){
    df <- data.frame(term = names(top_words_list[[i]]),
                     score = as.numeric(top_words_list[[i]]))
    print(
      ggplot(df, aes(x = reorder(term, score), y = score)) +
        geom_bar(stat="identity") +
        coord_flip() +
        ggtitle(paste0(title_prefix, " ", names(top_words_list)[i], " - Top 10 Words")) +
        theme_minimal()
    )
  }
}
plot_top_words(top_words_kmeans, "K-means")
plot_top_words(top_words_hc, "Hierarchical")


pca_res <- prcomp(tfidf_norm, center = TRUE, scale. = FALSE)

pca_df <- data.frame(
  PC1 = pca_res$x[,1],
  PC2 = pca_res$x[,2],
  label = labels,
  cluster = factor(kmeans_result$cluster)
)


ggplot(pca_df, aes(x = PC1, y = PC2, color = cluster, shape = label)) +
  geom_point(size = 3, alpha = 0.8) +
  ggtitle("K-means Clustering on TF-IDF (PCA Projection) - Normalized + Sparse Removed") +
  theme_minimal()

cat("\nPipeline complete.\n")

