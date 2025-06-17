import {
  VectorDocument,
  SearchResult,
  SimilarityMethod,
  SearchOptions,
  VectorStoreStats,
} from "./interfaces/vector-store.interface";

// OpenAI types
interface OpenAIEmbeddingResponse {
  data: Array<{
    embedding: number[];
    index: number;
  }>;
  model: string;
  usage: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

interface OpenAIInstance {
  embeddings: {
    create: (params: {
      input: string | string[];
      model: string;
    }) => Promise<OpenAIEmbeddingResponse>;
  };
}

interface NeuraDBOptions {
  openai?: OpenAIInstance;
  embeddingModel?: string;
  defaultBatchSize?: number;
  batchDelay?: number;
}

interface DocumentWithOptionalEmbedding extends Omit<VectorDocument, 'embedding'> {
  embedding?: number[];
}

interface AddDocumentsOptions {
  createEmbedding?: boolean;
  batchSize?: number;
  batchDelay?: number;
  onProgress?: (processed: number, total: number) => void;
}

interface SearchOptionsWithEmbedding extends Omit<SearchOptions, 'queryEmbedding'> {
  queryEmbedding?: number[];
}

/**
 * VectorStore provides high-performance in-memory vector similarity search
 * with support for multiple similarity methods, document management, and automatic OpenAI embeddings.
 * 
 * Features:
 * - Zero dependencies (OpenAI optional)
 * - Multiple similarity methods (cosine, euclidean, dot product)
 * - Automatic embedding generation with OpenAI
 * - Metadata filtering
 * - TypeScript support
 * - Memory-efficient storage
 * 
 * @example
 * ```typescript
 * import { NeuraDB } from 'vector-similarity-search';
 * import OpenAI from 'openai';
 * 
 * const openai = new OpenAI({ apiKey: 'your-api-key' });
 * const store = new NeuraDB({ openai });
 * 
 * // Add documents with automatic embedding generation
 * await store.addDocument({
 *   id: 'doc1',
 *   content: 'Hello world',
 *   metadata: { category: 'greeting' }
 * }, { createEmbedding: true });
 * 
 * // Search with automatic query embedding
 * const results = await store.search('Hello there', {
 *   limit: 5,
 *   threshold: 0.7,
 *   similarityMethod: 'cosine'
 * });
 * ```
 */
export class NeuraDB {
  private documents: Map<string, VectorDocument> = new Map();
  private openai?: OpenAIInstance;
  private embeddingModel: string;
  private defaultBatchSize: number = 10;
  private batchDelay: number = 1000;

  constructor(options: NeuraDBOptions = {}) {
    this.openai = options.openai;
    this.embeddingModel = options.embeddingModel || 'text-embedding-3-small';
    this.defaultBatchSize = options.defaultBatchSize || 100;
    this.batchDelay = options.batchDelay || 1000; // 1 second delay between batches
  }

  /**
   * Generate embedding using OpenAI
   * @param text Text to embed
   * @returns Embedding vector
   * @throws Error if OpenAI instance is not provided
   */
  async generateEmbedding(text: string): Promise<number[]> {
    if (!this.openai) {
      throw new Error('OpenAI instance is required for embedding generation. Please provide it in constructor.');
    }

    try {
      const response = await this.openai.embeddings.create({
        input: text,
        model: this.embeddingModel,
      });

      return response.data[0].embedding;
    } catch (error) {
      throw new Error(`Failed to generate embedding: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Generate embeddings for multiple texts using OpenAI with batch processing
   * @param texts Array of texts to embed
   * @param batchSize Number of texts to process in each batch
   * @param batchDelay Delay between batches in milliseconds
   * @returns Array of embedding vectors
   * @throws Error if OpenAI instance is not provided
   */
  async generateEmbeddings(
    texts: string[],
    batchSize: number = this.defaultBatchSize,
    batchDelay: number = this.batchDelay
  ): Promise<number[][]> {
    if (!this.openai) {
      throw new Error('OpenAI instance is required for embedding generation. Please provide it in constructor.');
    }

    if (texts.length === 0) {
      return [];
    }

    // For small batches, process all at once
    if (texts.length <= batchSize) {
      try {
        const response = await this.openai.embeddings.create({
          input: texts,
          model: this.embeddingModel,
        });

        return response.data
          .sort((a, b) => a.index - b.index)
          .map(item => item.embedding);
      } catch (error) {
        throw new Error(`Failed to generate embeddings: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    }

    // For large batches, process in chunks
    const allEmbeddings: number[][] = [];
    const batches = this.chunkArray(texts, batchSize);

    for (let i = 0; i < batches.length; i++) {
      const batch = batches[i];

      try {
        const response = await this.openai.embeddings.create({
          input: batch,
          model: this.embeddingModel,
        });

        const batchEmbeddings = response.data
          .sort((a, b) => a.index - b.index)
          .map(item => item.embedding);

        allEmbeddings.push(...batchEmbeddings);

        // Add delay between batches (except for the last batch)
        if (i < batches.length - 1 && batchDelay > 0) {
          await this.delay(batchDelay);
        }

      } catch (error) {
        throw new Error(`Failed to generate embeddings for batch ${i + 1}: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    }

    return allEmbeddings;
  }

  /**
   * Add a single document with pre-computed embedding or generate embedding automatically
   * @param document The document to add (embedding optional if createEmbedding is true)
   * @param options Options for document addition
   * @throws Error if document doesn't have a valid embedding or dimensions don't match
   * @example
   * ```typescript
   * // With pre-computed embedding
   * store.addDocument({
   *   id: 'doc1',
   *   content: 'Sample text',
   *   embedding: [0.1, 0.2, 0.3],
   *   metadata: { type: 'article' }
   * });
   * 
   * // With automatic embedding generation
   * await store.addDocument({
   *   id: 'doc2',
   *   content: 'Another text',
   *   metadata: { type: 'article' }
   * }, { createEmbedding: true });
   * ```
   */
  async addDocument(
    document: DocumentWithOptionalEmbedding,
    options: { createEmbedding?: boolean } = {}
  ): Promise<void> {
    let finalDocument: VectorDocument;

    if (options.createEmbedding && !document.embedding) {
      if (!document.content) {
        throw new Error('Document must have content when createEmbedding is true');
      }

      const embedding = await this.generateEmbedding(document.content);
      finalDocument = {
        ...document,
        embedding,
      } as VectorDocument;
    } else if (document.embedding) {
      finalDocument = document as VectorDocument;
    } else {
      throw new Error('Document must have an embedding or createEmbedding must be true');
    }

    this.validateDocument(finalDocument);

    // At this point finalDocument is validated and must have embedding
    if (!this.validateEmbeddingDimensions(finalDocument.embedding!)) {
      throw new Error(
        `Document embedding dimensions (${finalDocument.embedding!.length}) don't match existing documents (${this.getEmbeddingDimensions()})`
      );
    }

    const now = new Date();
    const documentWithTimestamps: VectorDocument = {
      ...finalDocument,
      createdAt: finalDocument.createdAt || now,
      updatedAt: now,
    };

    this.documents.set(finalDocument.id, documentWithTimestamps);
  }

  /**
   * Add multiple documents with pre-computed embeddings or generate embeddings automatically
   * @param documents Array of documents to add
   * @param options Options for document addition including batch processing
   * @throws Error if any document doesn't have a valid embedding
   * @example
   * ```typescript
   * // With automatic embedding generation and batch processing
   * await store.addDocuments([
   *   { id: '1', content: 'Text 1' },
   *   { id: '2', content: 'Text 2' }
   * ], { 
   *   createEmbedding: true,
   *   batchSize: 50,
   *   batchDelay: 500,
   *   onProgress: (processed, total) => console.log(`${processed}/${total}`)
   * });
   * ```
   */
  async addDocuments(
    documents: DocumentWithOptionalEmbedding[],
    options: AddDocumentsOptions = {}
  ): Promise<void> {
    const {
      createEmbedding = false,
      batchSize = this.defaultBatchSize,
      batchDelay = this.batchDelay,
      onProgress
    } = options;

    if (documents.length === 0) {
      return;
    }

    // Validate input documents early
    documents.forEach((doc, index) => {
      if (!doc.id || typeof doc.id !== 'string' || doc.id.trim() === '') {
        throw new Error(`Document at index ${index} must have a valid non-empty ID`);
      }

      // Check for duplicate IDs in the input array
      const duplicateIndex = documents.findIndex((otherDoc, otherIndex) =>
        otherIndex !== index && otherDoc.id === doc.id
      );
      if (duplicateIndex !== -1) {
        throw new Error(`Duplicate document ID '${doc.id}' found at indices ${index} and ${duplicateIndex}`);
      }

      // Check if document already exists in store
      if (this.documents.has(doc.id)) {
        throw new Error(`Document with ID '${doc.id}' already exists in store`);
      }

      if (createEmbedding) {
        if (!doc.embedding && (!doc.content || typeof doc.content !== 'string' || doc.content.trim() === '')) {
          throw new Error(
            `Document at index ${index} (ID: ${doc.id}) must have non-empty content when createEmbedding is true`
          );
        }
      } else {
        if (!doc.embedding || !Array.isArray(doc.embedding) || doc.embedding.length === 0) {
          throw new Error(
            `Document at index ${index} (ID: ${doc.id}) must have a valid embedding array when createEmbedding is false`
          );
        }

        // Validate embedding values
        if (doc.embedding.some(val => typeof val !== 'number' || !isFinite(val))) {
          throw new Error(
            `Document at index ${index} (ID: ${doc.id}) has invalid embedding values. All values must be finite numbers`
          );
        }

        // Check embedding dimensions consistency
        const expectedDim = this.getEmbeddingDimensions();
        if (expectedDim !== null && doc.embedding.length !== expectedDim) {
          throw new Error(
            `Document at index ${index} (ID: ${doc.id}) embedding dimensions (${doc.embedding.length}) don't match existing documents (${expectedDim})`
          );
        }
      }
    });

    let finalDocuments: VectorDocument[];

    if (createEmbedding) {
      // Separate documents that need embeddings vs those that already have them
      const documentsNeedingEmbeddings: DocumentWithOptionalEmbedding[] = [];
      const documentsWithEmbeddings: VectorDocument[] = [];

      documents.forEach((doc) => {
        if (!doc.embedding) {
          documentsNeedingEmbeddings.push(doc);
        } else {
          // Validate existing embedding
          if (!Array.isArray(doc.embedding) || doc.embedding.length === 0) {
            throw new Error(`Document ${doc.id} has invalid embedding array`);
          }

          if (doc.embedding.some(val => typeof val !== 'number' || !isFinite(val))) {
            throw new Error(`Document ${doc.id} has invalid embedding values`);
          }

          // Check dimensions
          const expectedDim = this.getEmbeddingDimensions();
          if (expectedDim !== null && doc.embedding.length !== expectedDim) {
            throw new Error(
              `Document ${doc.id} embedding dimensions (${doc.embedding.length}) don't match existing documents (${expectedDim})`
            );
          }

          documentsWithEmbeddings.push(doc as VectorDocument);
        }
      });

      // Generate embeddings for documents that need them
      let generatedEmbeddings: number[][] = [];
      if (documentsNeedingEmbeddings.length > 0) {
        const textsToEmbed = documentsNeedingEmbeddings.map(doc => doc.content!);

        try {
          if (onProgress) {
            // Process with progress tracking
            const batches = this.chunkArray(textsToEmbed, batchSize);
            let processedCount = 0;
            const totalToProcess = documentsNeedingEmbeddings.length;

            for (let i = 0; i < batches.length; i++) {
              const batch = batches[i];

              try {
                const batchEmbeddings = await this.generateEmbeddings(batch, batch.length, 0);
                generatedEmbeddings.push(...batchEmbeddings);

                processedCount += batch.length;
                onProgress(processedCount, totalToProcess);

                // Add delay between batches (except for the last batch)
                if (i < batches.length - 1 && batchDelay > 0) {
                  await this.delay(batchDelay);
                }
              } catch (error) {
                throw new Error(
                  `Failed to generate embeddings for batch ${i + 1} (documents ${processedCount + 1}-${processedCount + batch.length}): ${error instanceof Error ? error.message : 'Unknown error'
                  }`
                );
              }
            }
          } else {
            // Process without progress tracking
            generatedEmbeddings = await this.generateEmbeddings(textsToEmbed, batchSize, batchDelay);
          }
        } catch (error) {
          throw new Error(`Failed to generate embeddings: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
      }

      // Validate generated embeddings dimensions
      if (generatedEmbeddings.length > 0) {
        const expectedDim = this.getEmbeddingDimensions();
        const firstEmbeddingDim = generatedEmbeddings[0].length;

        if (expectedDim !== null && firstEmbeddingDim !== expectedDim) {
          throw new Error(
            `Generated embedding dimensions (${firstEmbeddingDim}) don't match existing documents (${expectedDim})`
          );
        }

        // Check all generated embeddings have consistent dimensions
        if (generatedEmbeddings.some(emb => emb.length !== firstEmbeddingDim)) {
          throw new Error('Generated embeddings have inconsistent dimensions');
        }
      }

      // Combine documents with their embeddings
      const docsWithNewEmbeddings = documentsNeedingEmbeddings.map((doc, index) => ({
        ...doc,
        embedding: generatedEmbeddings[index],
      })) as VectorDocument[];

      finalDocuments = [...documentsWithEmbeddings, ...docsWithNewEmbeddings];
    } else {
      // All documents should already have embeddings (validated above)
      finalDocuments = documents as VectorDocument[];
    }

    // Final validation and dimension consistency check
    const expectedDim = this.getEmbeddingDimensions();
    const now = new Date();

    // Check dimension consistency across all documents to be added
    if (finalDocuments.length > 0) {
      const firstDocDim = finalDocuments[0].embedding!.length;

      // Check consistency within the batch
      for (let i = 1; i < finalDocuments.length; i++) {
        if (finalDocuments[i].embedding!.length !== firstDocDim) {
          throw new Error(
            `Inconsistent embedding dimensions in batch: document ${finalDocuments[0].id} has ${firstDocDim} dimensions, but document ${finalDocuments[i].id} has ${finalDocuments[i].embedding!.length} dimensions`
          );
        }
      }

      // Check consistency with existing documents
      if (expectedDim !== null && firstDocDim !== expectedDim) {
        throw new Error(
          `Batch embedding dimensions (${firstDocDim}) don't match existing documents (${expectedDim})`
        );
      }
    }

    // Add all documents to the store
    try {
      finalDocuments.forEach(doc => {
        const documentWithTimestamps: VectorDocument = {
          ...doc,
          createdAt: doc.createdAt || now,
          updatedAt: now,
        };

        this.documents.set(doc.id, documentWithTimestamps);
      });
    } catch (error) {
      // Rollback any documents that were added before the error
      finalDocuments.forEach(doc => {
        this.documents.delete(doc.id);
      });

      throw new Error(`Failed to add documents to store: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Search for similar documents using vector similarity
   * @param query The query (can be embedding vector or text string)
   * @param options Search configuration options
   * @returns Array of search results sorted by similarity (highest first)
   * @throws Error if query is invalid
   * @example
   * ```typescript
   * // Search with pre-computed embedding
   * const results = store.search([0.1, 0.2, 0.3], {
   *   limit: 10,
   *   threshold: 0.5,
   *   similarityMethod: 'cosine',
   *   metadataFilter: { category: 'news' }
   * });
   * 
   * // Search with text query (automatic embedding)
   * const results = await store.search('Hello world', {
   *   limit: 10,
   *   threshold: 0.5,
   *   similarityMethod: 'cosine'
   * });
   * ```
   */
  async search(
    query: number[] | string,
    options: SearchOptions = {}
  ): Promise<SearchResult[]> {
    const {
      limit = 10,
      threshold = 0,
      similarityMethod = "cosine",
      metadataFilter,
      page = 1,
      pageSize,
    } = options;

    if (this.documents.size === 0) {
      return [];
    }

    let queryEmbedding: number[];

    // Handle different query types
    if (typeof query === 'string') {
      // Generate embedding for text query
      queryEmbedding = await this.generateEmbedding(query);
    } else if (Array.isArray(query)) {
      // Use provided embedding
      if (query.length === 0) {
        throw new Error("Query embedding must be provided and non-empty");
      }
      queryEmbedding = query;
    } else {
      throw new Error("Query must be either a string or number array");
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
          document.embedding!,
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

    // Sort by similarity (highest first)
    const sortedResults = results.sort((a, b) => b.similarity - a.similarity);

    // Pagination logic
    let paginatedResults = sortedResults;
    if (pageSize !== undefined) {
      const startIdx = (page - 1) * pageSize;
      paginatedResults = sortedResults.slice(startIdx, startIdx + pageSize);
    } else {
      paginatedResults = sortedResults.slice(0, limit);
    }

    return paginatedResults;
  }

  /**
   * Find the most similar document to the query
   * @param query The query (can be embedding vector or text string)
   * @param similarityMethod Similarity calculation method
   * @returns Most similar document or null if none found
   * @example
   * ```typescript
   * const mostSimilar = await store.findMostSimilar('Hello world', 'cosine');
   * if (mostSimilar) {
   *   console.log(`Most similar: ${mostSimilar.document.content}`);
   * }
   * ```
   */
  async findMostSimilar(
    query: number[] | string,
    similarityMethod: SimilarityMethod = "cosine"
  ): Promise<SearchResult | null> {
    const results = await this.search(query, { limit: 1, similarityMethod });
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
   * @param document Updated document (embedding optional if createEmbedding is true)
   * @param options Options for document update
   * @returns True if document was updated, false if not found
   * @throws Error if document doesn't have a valid embedding
   */
  async updateDocument(
    document: DocumentWithOptionalEmbedding,
    options: { createEmbedding?: boolean } = {}
  ): Promise<boolean> {
    if (!this.documents.has(document.id)) {
      return false;
    }

    let finalDocument: VectorDocument;

    if (options.createEmbedding && !document.embedding) {
      if (!document.content) {
        throw new Error('Document must have content when createEmbedding is true');
      }

      const embedding = await this.generateEmbedding(document.content);
      finalDocument = {
        ...document,
        embedding,
      } as VectorDocument;
    } else if (document.embedding) {
      finalDocument = document as VectorDocument;
    } else {
      throw new Error('Document must have an embedding or createEmbedding must be true');
    }

    this.validateDocument(finalDocument);

    if (!this.validateEmbeddingDimensions(finalDocument.embedding!)) {
      throw new Error(
        `Document embedding dimensions (${finalDocument.embedding!.length}) don't match existing documents (${this.getEmbeddingDimensions()})`
      );
    }

    const existingDoc = this.documents.get(finalDocument.id)!;
    const updatedDocument: VectorDocument = {
      ...finalDocument,
      createdAt: existingDoc.createdAt,
      updatedAt: new Date(),
    };

    this.documents.set(finalDocument.id, updatedDocument);
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
    return firstDoc ? firstDoc.embedding!.length : null;
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
      estimatedMemoryUsage += doc.embedding!.length * 8; // 8 bytes per number
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

  /**
   * Check if OpenAI instance is available
   * @returns True if OpenAI instance is configured
   */
  hasOpenAI(): boolean {
    return !!this.openai;
  }

  /**
   * Get current embedding model name
   * @returns Embedding model name
   */
  getEmbeddingModel(): string {
    return this.embeddingModel;
  }

  /**
   * Set embedding model name
   * @param model New embedding model name
   */
  setEmbeddingModel(model: string): void {
    this.embeddingModel = model;
  }

  /**
   * Get default batch size for embedding operations
   * @returns Default batch size
   */
  getDefaultBatchSize(): number {
    return this.defaultBatchSize;
  }

  /**
   * Set default batch size for embedding operations
   * @param batchSize New default batch size
   */
  setDefaultBatchSize(batchSize: number): void {
    if (batchSize <= 0) {
      throw new Error('Batch size must be greater than 0');
    }
    this.defaultBatchSize = batchSize;
  }

  /**
   * Get default batch delay
   * @returns Default batch delay in milliseconds
   */
  getDefaultBatchDelay(): number {
    return this.batchDelay;
  }

  /**
   * Set default batch delay
   * @param delay New default batch delay in milliseconds
   */
  setDefaultBatchDelay(delay: number): void {
    if (delay < 0) {
      throw new Error('Batch delay cannot be negative');
    }
    this.batchDelay = delay;
  }

  // Private methods

  /**
   * Split array into chunks of specified size
   */
  private chunkArray<T>(array: T[], chunkSize: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += chunkSize) {
      chunks.push(array.slice(i, i + chunkSize));
    }
    return chunks;
  }

  /**
   * Delay execution for specified milliseconds
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Validate document structure and embedding
   */
  private validateDocument(document: VectorDocument): boolean {
    if (!document.id) {
      throw new Error("Document must have an ID");
    }

    if (!document.embedding || document.embedding.length === 0) {
      throw new Error("Document must have a valid embedding");
    }

    if (document.embedding.some(val => typeof val !== 'number' || !isFinite(val))) {
      throw new Error("All embedding values must be finite numbers");
    }

    return true;
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

  /**
   * Search for similar documents with pagination metadata
   * @param query The query (can be embedding vector or text string)
   * @param options Search configuration options (must include pageSize)
   * @returns Object with data (results), page, pageSize, totalResults, totalPages
   */
  async searchWithPagination(
    query: number[] | string,
    options: SearchOptions = {}
  ): Promise<{
    data: SearchResult[];
    page: number;
    pageSize: number;
    totalResults: number;
    totalPages: number;
  }> {
    const {
      page = 1,
      pageSize = 10,
      threshold = 0,
      similarityMethod = "cosine",
      metadataFilter,
    } = options;

    if (this.documents.size === 0) {
      return {
        data: [],
        page,
        pageSize,
        totalResults: 0,
        totalPages: 0,
      };
    }

    let queryEmbedding: number[];
    if (typeof query === 'string') {
      queryEmbedding = await this.generateEmbedding(query);
    } else if (Array.isArray(query)) {
      if (query.length === 0) {
        throw new Error("Query embedding must be provided and non-empty");
      }
      queryEmbedding = query;
    } else {
      throw new Error("Query must be either a string or number array");
    }

    let documentsToSearch = Array.from(this.documents.values());
    if (metadataFilter) {
      documentsToSearch = this.filterByMetadata(documentsToSearch, metadataFilter);
    }

    const results: SearchResult[] = [];
    for (const document of documentsToSearch) {
      try {
        const similarity = this.calculateSimilarity(
          queryEmbedding,
          document.embedding!,
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

    // Sort by similarity (highest first)
    const sortedResults = results.sort((a, b) => b.similarity - a.similarity);
    const totalResults = sortedResults.length;
    const totalPages = Math.ceil(totalResults / pageSize);
    const startIdx = (page - 1) * pageSize;
    const data = sortedResults.slice(startIdx, startIdx + pageSize);

    return {
      data,
      page,
      pageSize,
      totalResults,
      totalPages,
    };
  }
}