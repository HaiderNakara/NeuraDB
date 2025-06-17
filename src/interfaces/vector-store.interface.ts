/**
 * Represents a document with its vector embedding and metadata
 */
export interface VectorDocument {
  /** Unique identifier for the document */
  id: string;

  /** The text content of the document */
  content: string;

  /** Vector embedding representation of the document */
  embedding?: number[];

  /** Optional metadata associated with the document */
  metadata?: Record<string, any>;

  /** Optional timestamp when the document was created */
  createdAt?: Date;

  /** Optional timestamp when the document was last updated */
  updatedAt?: Date;
}

/**
 * Result of a vector similarity search
 */
export interface SearchResult {
  /** The document that matched the search */
  document: VectorDocument;

  /** Similarity score between 0 and 1 (1 being most similar) */
  similarity: number;
}

/**
 * Supported similarity calculation methods
 */
export type SimilarityMethod = "cosine" | "euclidean" | "dot";

/**
 * Configuration options for vector search
 */
export interface SearchOptions {
  /** Maximum number of results to return */
  limit?: number;

  /** Minimum similarity threshold (0-1) */
  threshold?: number;

  /** Similarity calculation method to use */
  similarityMethod?: SimilarityMethod;

  /** Metadata filters to apply */
  metadataFilter?: Record<string, any>;

  /** Page number for pagination (1-based) */
  page?: number;

  /** Page size for pagination */
  pageSize?: number;
}

/**
 * Statistics about the vector store
 */
export interface VectorStoreStats {
  /** Total number of documents stored */
  documentCount: number;

  /** Dimensions of the vector embeddings */
  embeddingDimensions: number | null;

  /** Memory usage estimation in bytes */
  estimatedMemoryUsage: number;
}