import {
  VectorDocument,
  SearchResult,
  SimilarityMethod,
  SearchOptions,
  VectorStoreStats,
} from "./interfaces/vector-store.interface";

/**
 * VectorStore provides high-performance in-memory vector similarity search
 * with support for multiple similarity methods and document management.
 * 
 * Features:
 * - Zero dependencies
 * - Multiple similarity methods (cosine, euclidean, dot product)
 * - Metadata filtering
 * - TypeScript support
 * - Memory-efficient storage
 * 
 * @example
 * ```typescript
 * import { VectorStore } from 'vector-similarity-search';
 * 
 * const store = new VectorStore();
 * 
 * // Add documents with embeddings
 * store.addDocument({
 *   id: 'doc1',
 *   content: 'Hello world',
 *   embedding: [0.1, 0.2, 0.3],
 *   metadata: { category: 'greeting' }
 * });
 * 
 * // Search for similar documents
 * const results = store.search([0.1, 0.2, 0.3], {
 *   limit: 5,
 *   threshold: 0.7,
 *   similarityMethod: 'cosine'
 * });
 * ```
 */
export class VectorStore {
  private documents: Map<string, VectorDocument> = new Map();

  /**
   * Add a single document with pre-computed embedding
   * @param document The document to add
   * @throws Error if document doesn't have a valid embedding or dimensions don't match
   * @example
   * ```typescript
   * store.addDocument({
   *   id: 'doc1',
   *   content: 'Sample text',
   *   embedding: [0.1, 0.2, 0.3],
   *   metadata: { type: 'article' }
   * });
   * ```
   */
  addDocument(document: VectorDocument): void {
    this.validateDocument(document);

    if (!this.validateEmbeddingDimensions(document.embedding)) {
      throw new Error(
        `Document embedding dimensions (${document.embedding.length}) don't match existing documents (${this.getEmbeddingDimensions()})`
      );
    }

    const now = new Date();
    const documentWithTimestamps: VectorDocument = {
      ...document,
      createdAt: document.createdAt || now,
      updatedAt: now,
    };

    this.documents.set(document.id, documentWithTimestamps);
  }

  /**
   * Add multiple documents with pre-computed embeddings
   * @param documents Array of documents to add
   * @throws Error if any document doesn't have a valid embedding
   * @example
   * ```typescript
   * store.addDocuments([
   *   { id: '1', content: 'Text 1', embedding: [0.1, 0.2] },
   *   { id: '2', content: 'Text 2', embedding: [0.3, 0.4] }
   * ]);
   * ```
   */
  addDocuments(documents: VectorDocument[]): void {
    // Validate all documents first
    documents.forEach((doc) => this.validateDocument(doc));

    // Add all documents
    documents.forEach((doc) => this.addDocument(doc));
  }

  /**
   * Search for similar documents using vector similarity
   * @param queryEmbedding The query vector embedding
   * @param options Search configuration options
   * @returns Array of search results sorted by similarity (highest first)
   * @throws Error if query embedding is invalid
   * @example
   * ```typescript
   * const results = store.search([0.1, 0.2, 0.3], {
   *   limit: 10,
   *   threshold: 0.5,
   *   similarityMethod: 'cosine',
   *   metadataFilter: { category: 'news' }
   * });
   * ```
   */
  search(
    queryEmbedding: number[],
    options: SearchOptions = {}
  ): SearchResult[] {
    const {
      limit = 10,
      threshold = 0,
      similarityMethod = "cosine",
      metadataFilter,
    } = options;

    if (this.documents.size === 0) {
      throw new Error("Query embedding must be provided and non-empty");
    }

    if (!queryEmbedding || queryEmbedding.length === 0) {
      throw new Error("Query embedding must be provided and non-empty");
    }

    const results: SearchResult[] = [];
    let documentsToSearch = Array.from(this.documents.values());

    // Apply metadata filter if provided
    if (metadataFilter) {
      documentsToSearch = this.filterByMetadata(documentsToSearch, metadataFilter);
    }

    for (const document of documentsToSearch) {
      try {
        const similarity = this.calculateSimilarity(
          queryEmbedding,
          document.embedding,
          similarityMethod
        );

        if (similarity >= threshold) {
          results.push({ document, similarity });
        }
      } catch (error) {
        console.warn(`Skipping document ${document.id}: ${error}`);
        continue;
      }
    }

    // Sort by similarity (highest first) and limit results
    return results
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit);
  }

  /**
   * Find the most similar document to the query
   * @param queryEmbedding The query vector embedding
   * @param similarityMethod Similarity calculation method
   * @returns Most similar document or null if none found
   * @example
   * ```typescript
   * const mostSimilar = store.findMostSimilar([0.1, 0.2, 0.3], 'cosine');
   * if (mostSimilar) {
   *   console.log(`Most similar: ${mostSimilar.document.content}`);
   * }
   * ```
   */
  findMostSimilar(
    queryEmbedding: number[],
    similarityMethod: SimilarityMethod = "cosine"
  ): SearchResult | null {
    const results = this.search(queryEmbedding, { limit: 1, similarityMethod });
    return results.length > 0 ? results[0] : null;
  }

  /**
   * Get document by ID
   * @param id Document ID
   * @returns Document or undefined if not found
   */
  getDocument(id: string): VectorDocument | undefined {
    return this.documents.get(id);
  }

  /**
   * Check if a document exists
   * @param id Document ID
   * @returns True if document exists
   */
  hasDocument(id: string): boolean {
    return this.documents.has(id);
  }

  /**
   * Remove document by ID
   * @param id Document ID to remove
   * @returns True if document was removed, false if not found
   */
  removeDocument(id: string): boolean {
    return this.documents.delete(id);
  }

  /**
   * Update an existing document
   * @param document Updated document
   * @returns True if document was updated, false if not found
   * @throws Error if document doesn't have a valid embedding
   */
  updateDocument(document: VectorDocument): boolean {
    if (!this.documents.has(document.id)) {
      return false;
    }

    this.validateDocument(document);

    if (!this.validateEmbeddingDimensions(document.embedding)) {
      throw new Error(
        `Document embedding dimensions (${document.embedding.length}) don't match existing documents (${this.getEmbeddingDimensions()})`
      );
    }

    const existingDoc = this.documents.get(document.id)!;
    const updatedDocument: VectorDocument = {
      ...document,
      createdAt: existingDoc.createdAt,
      updatedAt: new Date(),
    };

    this.documents.set(document.id, updatedDocument);
    return true;
  }

  /**
   * Get all documents
   * @returns Array of all documents
   */
  getAllDocuments(): VectorDocument[] {
    return Array.from(this.documents.values());
  }

  /**
   * Get documents by metadata filter
   * @param filter Metadata filter criteria
   * @returns Array of matching documents
   * @example
   * ```typescript
   * const newsArticles = store.getDocumentsByMetadata({ category: 'news' });
   * ```
   */
  getDocumentsByMetadata(filter: Record<string, any>): VectorDocument[] {
    return this.filterByMetadata(this.getAllDocuments(), filter);
  }

  /**
   * Clear all documents from the store
   */
  clear(): void {
    this.documents.clear();
  }

  /**
   * Get the number of documents in the store
   * @returns Number of documents stored
   */
  size(): number {
    return this.documents.size;
  }

  /**
   * Check if the store is empty
   * @returns True if no documents are stored
   */
  isEmpty(): boolean {
    return this.documents.size === 0;
  }

  /**
   * Get embedding dimensions from stored documents
   * @returns Number of dimensions or null if no documents
   */
  getEmbeddingDimensions(): number | null {
    const firstDoc = Array.from(this.documents.values())[0];
    return firstDoc ? firstDoc.embedding.length : null;
  }

  /**
   * Get comprehensive statistics about the vector store
   * @returns Statistics including document count, dimensions, and memory usage
   */
  getStats(): VectorStoreStats {
    const documentCount = this.size();
    const embeddingDimensions = this.getEmbeddingDimensions();

    // Estimate memory usage
    let estimatedMemoryUsage = 0;
    for (const doc of this.documents.values()) {
      estimatedMemoryUsage += doc.embedding.length * 8; // 8 bytes per number
      estimatedMemoryUsage += doc.content.length * 2; // 2 bytes per character (UTF-16)
      estimatedMemoryUsage += JSON.stringify(doc.metadata || {}).length * 2;
      estimatedMemoryUsage += 100; // Overhead for object structure
    }

    return {
      documentCount,
      embeddingDimensions,
      estimatedMemoryUsage,
    };
  }

  // Private methods

  /**
   * Validate document structure and embedding
   */
  private validateDocument(document: VectorDocument): void {
    if (!document.id) {
      throw new Error("Document must have an ID");
    }

    if (!document.embedding || document.embedding.length === 0) {
      throw new Error("Document must have a valid embedding");
    }

    if (document.embedding.some(val => typeof val !== 'number' || !isFinite(val))) {
      throw new Error("All embedding values must be finite numbers");
    }
  }

  /**
   * Validate embedding dimensions against existing documents
   */
  private validateEmbeddingDimensions(embedding: number[]): boolean {
    const expectedDim = this.getEmbeddingDimensions();
    if (expectedDim === null) return true; // No documents yet
    return embedding.length === expectedDim;
  }

  /**
   * Filter documents by metadata criteria
   */
  private filterByMetadata(
    documents: VectorDocument[],
    filter: Record<string, any>
  ): VectorDocument[] {
    return documents.filter((doc) => {
      if (!doc.metadata) return false;

      return Object.entries(filter).every(
        ([key, value]) => doc.metadata![key] === value
      );
    });
  }

  /**
   * Calculate similarity between two vectors using specified method
   */
  private calculateSimilarity(
    vecA: number[],
    vecB: number[],
    method: SimilarityMethod
  ): number {
    if (vecA.length !== vecB.length) {
      throw new Error("Vectors must have the same dimensions");
    }

    switch (method) {
      case "cosine":
        return this.cosineSimilarity(vecA, vecB);
      case "euclidean":
        return this.euclideanSimilarity(vecA, vecB);
      case "dot":
        return this.dotProductSimilarity(vecA, vecB);
      default:
        return this.cosineSimilarity(vecA, vecB);
    }
  }

  /**
   * Calculate cosine similarity between two vectors
   * @returns Similarity score between -1 and 1 (1 being identical)
   */
  private cosineSimilarity(vecA: number[], vecB: number[]): number {
    const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
    const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));

    if (magnitudeA === 0 || magnitudeB === 0) {
      return 0;
    }

    return dotProduct / (magnitudeA * magnitudeB);
  }

  /**
   * Calculate Euclidean similarity between two vectors
   * @returns Similarity score between 0 and 1 (1 being identical)
   */
  private euclideanSimilarity(vecA: number[], vecB: number[]): number {
    const distance = Math.sqrt(
      vecA.reduce((sum, a, i) => sum + Math.pow(a - vecB[i], 2), 0)
    );

    // Convert distance to similarity score (0-1, where 1 is most similar)
    return 1 / (1 + distance);
  }

  /**
   * Calculate dot product similarity between two vectors
   * @returns Dot product value
   */
  private dotProductSimilarity(vecA: number[], vecB: number[]): number {
    return vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  }
}