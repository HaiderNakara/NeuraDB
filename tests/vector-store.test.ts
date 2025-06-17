import { VectorDocument } from '../src/interfaces/vector-store.interface';
import { NeuraDB } from '../src/vector-store';

describe('VectorStore', () => {
  let store: NeuraDB;
  const sampleDocument: VectorDocument = {
    id: 'doc1',
    content: 'Hello world',
    embedding: [0.1, 0.2, 0.3],
    metadata: { category: 'greeting' }
  };

  beforeEach(() => {
    store = new NeuraDB();
  });

  describe('Document Management', () => {
    test('should add and retrieve a document', () => {
      store.addDocument(sampleDocument);
      const retrieved = store.getDocument('doc1');
      expect(retrieved).toEqual(expect.objectContaining(sampleDocument));
    });

    test('should add multiple documents', () => {
      const docs = [
        sampleDocument,
        { ...sampleDocument, id: 'doc2', content: 'Hello again' }
      ];
      store.addDocuments(docs);
      expect(store.size()).toBe(2);
    });

    test('should update an existing document', async () => {
      store.addDocument(sampleDocument);
      const updated = { ...sampleDocument, content: 'Updated content' };
      const success = await store.updateDocument(updated);
      expect(success).toBe(true);
      expect(store.getDocument('doc1')?.content).toBe('Updated content');
    });

    test('should remove a document', () => {
      store.addDocument(sampleDocument);
      const removed = store.removeDocument('doc1');
      expect(removed).toBe(true);
      expect(store.hasDocument('doc1')).toBe(false);
    });

    test('should validate document embedding dimensions', async () => {
      await store.addDocument(sampleDocument);
      const invalidDoc = {
        ...sampleDocument,
        id: 'doc2',
        embedding: [0.1, 0.2] // Different dimensions
      };

      await expect(store.addDocument(invalidDoc)).rejects.toThrow("Document embedding dimensions (2) don't match existing documents (3)");
    });
  });

  describe('Search Operations', () => {
    beforeEach(() => {
      store.addDocument(sampleDocument);
      store.addDocument({
        ...sampleDocument,
        id: 'doc2',
        content: 'Different content',
        embedding: [0.2, 0.3, 0.4]
      });
      // Add more documents for pagination
      for (let i = 3; i <= 7; i++) {
        store.addDocument({
          ...sampleDocument,
          id: `doc${i}`,
          content: `Content ${i}`,
          embedding: [0.1 * i, 0.2 * i, 0.3 * i],
        });
      }
    });

    test('should find most similar document', async () => {
      const result = await store.findMostSimilar([0.1, 0.2, 0.3]);
      expect(result).toBeDefined();
      expect(result?.document.id).toBe('doc1');
    });

    test('should filter by metadata', async () => {
      console.log(store.getAllDocuments());

      const results = await store.search([0.1, 0.2, 0.3], {
        metadataFilter: { category: 'greeting' },
        page: 2,
        pageSize: 2
      });
      expect(results.length).toBe(2);
      expect(results[0].document.metadata?.category).toBe('greeting');
    });

    test('should respect search limit', async () => {
      const results = await store.search([0.1, 0.2, 0.3], { limit: 1 });
      expect(results.length).toBe(1);
    });

    test('should respect similarity threshold', async () => {
      const results = await store.search([1, 1, 1], { threshold: 0.99 });
      expect(results.length).toBe(0);
    });

    test('should paginate search results', async () => {
      // All docs are similar, so all will be returned if no limit
      const pageSize = 2;
      // Page 1
      let results = await store.search([0.1, 0.2, 0.3], { page: 1, pageSize });
      expect(results.length).toBe(2);
      expect(results[0].document.id).toBeDefined();
      // Page 2
      results = await store.search([0.1, 0.2, 0.3], { page: 2, pageSize });
      expect(results.length).toBe(2);
      // Page 3
      results = await store.search([0.1, 0.2, 0.3], { page: 3, pageSize });
      expect(results.length).toBe(2);
      // Page 4 (should have 1 doc left)
      results = await store.search([0.1, 0.2, 0.3], { page: 4, pageSize });
      expect(results.length).toBe(1);
      // Page 5 (should be empty)
      results = await store.search([0.1, 0.2, 0.3], { page: 5, pageSize });
      expect(results.length).toBe(0);
    });
  });

  describe('Similarity Methods', () => {
    const vecA = [1, 0, 0];
    const vecB = [0, 1, 0];

    test('should use cosine similarity by default', async () => {
      store.addDocument({ ...sampleDocument, embedding: vecA });
      const result = await store.search(vecB);
      expect(result[0].similarity).toBeLessThan(1);
    });

    test('should support euclidean similarity', async () => {
      store.addDocument({ ...sampleDocument, embedding: vecA });
      const result = await store.search(vecB, { similarityMethod: 'euclidean' });
      expect(result[0].similarity).toBeGreaterThan(0);
    });

    test('should support dot product similarity', async () => {
      store.addDocument({ ...sampleDocument, embedding: vecA });
      const result = await store.search(vecB, { similarityMethod: 'dot' });
      expect(result[0].similarity).toBe(0);
    });
  });

  describe('Store Statistics', () => {
    test('should return correct stats', () => {
      store.addDocument(sampleDocument);
      const stats = store.getStats();
      expect(stats.documentCount).toBe(1);
      expect(stats.embeddingDimensions).toBe(3);
      expect(stats.estimatedMemoryUsage).toBeGreaterThan(0);
    });

    test('should handle empty store', () => {
      const stats = store.getStats();
      expect(stats.documentCount).toBe(0);
      expect(stats.embeddingDimensions).toBeNull();
      expect(stats.estimatedMemoryUsage).toBe(0);
    });
  });

  describe('Error Handling', () => {
    test('should throw on invalid document ID', () => {
      expect(store.addDocument({ ...sampleDocument, id: '' }))
        .rejects.toThrow('Document must have an ID');
    });

    test('should throw on empty embedding', () => {
      expect(store.addDocument({ ...sampleDocument, embedding: [] }))
        .rejects.toThrow('Document must have a valid embedding');
    });

    test('should throw on invalid embedding values', () => {
      expect(store.addDocument({ ...sampleDocument, embedding: [NaN, 0.2, 0.3] }))
        .rejects.toThrow('All embedding values must be finite numbers');
    });

    test('should throw on invalid query embedding', async () => {
      // First add a document to ensure the store is not empty
      await store.addDocument(sampleDocument);
      // Now test with empty embedding
      expect(store.search([]))
        .rejects.toThrow('Query embedding must be provided and non-empty');
    });
  });

  describe('Document Addition with Embeddings', () => {
    const mockOpenAI = {
      embeddings: {
        create: jest.fn().mockImplementation(async ({ input }) => {
          // Handle both single string and array of strings
          const inputs = Array.isArray(input) ? input : [input];
          return {
            data: inputs.map((_, index) => ({
              embedding: [0.1, 0.2, 0.3],
              index
            })),
            model: 'text-embedding-3-small',
            usage: { prompt_tokens: 0, total_tokens: 0 }
          };
        })
      }
    };

    beforeEach(() => {
      store = new NeuraDB({ openai: mockOpenAI });
      mockOpenAI.embeddings.create.mockClear();
    });

    test('should add documents with createEmbedding option', async () => {
      const docs = [
        { id: 'doc1', content: 'Hello world', metadata: { category: 'greeting' } },
        { id: 'doc2', content: 'Hello again', metadata: { category: 'greeting' } }
      ];

      await store.addDocuments(docs, { createEmbedding: true });

      expect(store.size()).toBe(2);
      expect(store.getDocument('doc1')).toBeDefined();
      expect(store.getDocument('doc2')).toBeDefined();
      expect(mockOpenAI.embeddings.create).toHaveBeenCalledTimes(1);
      expect(mockOpenAI.embeddings.create).toHaveBeenCalledWith({
        input: ['Hello world', 'Hello again'],
        model: 'text-embedding-3-small'
      });
    });

    test('should handle mixed documents with and without embeddings', async () => {
      const docs = [
        { id: 'doc1', content: 'Hello world', metadata: { category: 'greeting' } },
        { id: 'doc2', content: 'Hello again', embedding: [0.1, 0.2, 0.3], metadata: { category: 'greeting' } }
      ];

      await store.addDocuments(docs, { createEmbedding: true });

      expect(store.size()).toBe(2);
      expect(store.getDocument('doc1')).toBeDefined();
      expect(store.getDocument('doc2')).toBeDefined();
      expect(mockOpenAI.embeddings.create).toHaveBeenCalledTimes(1);
      expect(mockOpenAI.embeddings.create).toHaveBeenCalledWith({
        input: ['Hello world'],
        model: 'text-embedding-3-small'
      });
    });

    test('should throw error when document has no content for embedding', async () => {
      const docs = [
        { id: 'doc1', metadata: { category: 'greeting' } } as any
      ];

      await expect(store.addDocuments(docs, { createEmbedding: true }))
        .rejects.toThrow('Document at index 0 (ID: doc1) must have non-empty content when createEmbedding is true');
    });

    test('should handle progress callback', async () => {
      const docs = [
        { id: 'doc1', content: 'Hello world' },
        { id: 'doc2', content: 'Hello again' }
      ];

      const progressCallback = jest.fn();
      await store.addDocuments(docs, {
        createEmbedding: true,
        onProgress: progressCallback
      });

      expect(progressCallback).toHaveBeenCalled();
      expect(store.size()).toBe(2);
    });

    test('should validate embedding dimensions after generation', async () => {
      // First add a document to set the dimension
      await store.addDocument(sampleDocument);

      // Try to add a document with different dimensions
      const docs = [
        { id: 'doc2', content: 'Hello world' }
      ];

      // Mock OpenAI to return different dimensions
      mockOpenAI.embeddings.create.mockResolvedValueOnce({
        data: [{ embedding: [0.1, 0.2], index: 0 }],
        model: 'text-embedding-3-small',
        usage: { prompt_tokens: 0, total_tokens: 0 }
      });

      await expect(store.addDocuments(docs, { createEmbedding: true }))
        .rejects.toThrow("Generated embedding dimensions (2) don't match existing documents (3)");
    });

    test('should handle batch processing with delay', async () => {
      const docs = Array.from({ length: 15 }, (_, i) => ({
        id: `doc${i + 1}`,
        content: `Content ${i + 1}`
      }));

      const startTime = Date.now();
      await store.addDocuments(docs, {
        createEmbedding: true,
        batchSize: 5,
        batchDelay: 100
      });
      const endTime = Date.now();

      expect(store.size()).toBe(15);
      // Should be called 3 times (3 batches of 5)
      expect(mockOpenAI.embeddings.create).toHaveBeenCalledTimes(3);
      // Should have at least 2 delays (3 batches of 5)
      expect(endTime - startTime).toBeGreaterThanOrEqual(200);
    });

    test('should throw error on duplicate document IDs', async () => {
      const docs = [
        { id: 'doc1', content: 'Hello world' },
        { id: 'doc1', content: 'Hello again' }
      ];

      await expect(store.addDocuments(docs, { createEmbedding: true }))
        .rejects.toThrow("Duplicate document ID 'doc1' found at indices 0 and 1");
    });

    test('should throw error on empty document ID', async () => {
      const docs = [
        { id: '', content: 'Hello world' } as any
      ];

      await expect(store.addDocuments(docs, { createEmbedding: true }))
        .rejects.toThrow('Document at index 0 must have a valid non-empty ID');
    });

    test('should throw error on invalid embedding values', async () => {
      const docs = [
        { id: 'doc1', content: 'Hello world', embedding: [NaN, 0.2, 0.3] }
      ];

      await expect(store.addDocuments(docs))
        .rejects.toThrow('Document at index 0 (ID: doc1) has invalid embedding values. All values must be finite numbers');
    });
  });
}); 