

library(spreadr)
library(igraph)
#library(tidyverse)

# define the spreadr function that takes:
# a graph, a word list, and an initial activation level of a node
spreadSimple <- function(g, word_list) {
  # get the node names
  nodes <- V(g)$name
  # set the time steps based on the diameter
  t <- ceiling(2*diameter(g, weights = NA))
  # create a results df to store the data with node and time step
  df <- data.frame("node" = nodes, "time" = sort(rep(0:t, length(nodes))))
  df_final <- data.frame("node" = nodes)
  # loop through the words
  for (i in seq_along(word_list)){
    word <- word_list[i]
    # create the initial activation data frame
    start_run <- data.frame(node = nodes, activation = rep(0, length(nodes)))
    # set the total activation level based on the # of nodes
    act <- length(nodes)
    start_run[which(start_run$node == word), 2] <- act
    # convert graph to adjacency matrix otherwise the weights are not considered
    if (is_weighted(g)){
      adj <- as_adjacency_matrix(g, sparse = FALSE,  attr = "weight")
    } else {
      adj <- as_adjacency_matrix(g, sparse = FALSE)
    }
    # run the spreading process
    result <- spreadr(adj, start_run, time = t, include_t0 = TRUE)
    # add a column in the results df for that activated node set
    df$act_level <- result$activation
    final_act <- df$act_level[which(df$time == max(df$time))]
    df_final[[word]] <- final_act
  }
  # The resulting df has columns corresponding to the node initially activated
  # Rows correspond to the activation level of each node at each time step
  return(df_final)
}

# Full graphs
#humans <- read_graph("FA_Humans_ig.graphml", format = "graphml")
#mistral <- read_graph("FA_Mistral_ig.graphml", format = "graphml")
llama3 <- read_graph("FA_Llama3_ig.graphml", format = "graphml")
#haiku <- read_graph("FA_Haiku_ig.graphml", format = "graphml")
#gpt35 <- read_graph("FA_GPT-3.5 Turbo_ig.graphml", format = "graphml")
#gpt4o <- read_graph("FA_GPT-4o_ig.graphml", format = "graphml")
#gpt4t <- read_graph("FA_GPT-4 Turbo_ig.graphml", format = "graphml")

primeList <- read.csv("primes.csv", header = TRUE)
primes <- as.character(primeList$prime)

genderList <- read.csv("genderWords.csv", header = TRUE)
gender_words <- as.character(genderList$genderWord)

df <- spreadSimple(llama3, primes)
write.csv(df, paste0("Llama3_primes.csv"), row.names = FALSE)

df <- spreadSimple(llama3, gender_words)
write.csv(df, paste0("Llama3_gender.csv"), row.names = FALSE)

