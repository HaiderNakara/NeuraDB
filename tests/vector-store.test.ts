import { VectorStore } from '../src/vector-store';
import { VectorDocument, SimilarityMethod } from '../src/interfaces/vector-store.interface';

describe('VectorStore', () => {
  let store: VectorStore;
  const sampleDocument: VectorDocument = {
    id: 'doc1',
    content: 'Hello world',
    embedding: [0.1, 0.2, 0.3],
    metadata: { category: 'greeting' }
  };

  beforeEach(() => {
    store = new VectorStore();
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

    test('should update an existing document', () => {
      store.addDocument(sampleDocument);
      const updated = { ...sampleDocument, content: 'Updated content' };
      const success = store.updateDocument(updated);
      expect(success).toBe(true);
      expect(store.getDocument('doc1')?.content).toBe('Updated content');
    });

    test('should remove a document', () => {
      store.addDocument(sampleDocument);
      const removed = store.removeDocument('doc1');
      expect(removed).toBe(true);
      expect(store.hasDocument('doc1')).toBe(false);
    });

    test('should validate document embedding dimensions', () => {
      store.addDocument(sampleDocument);
      const invalidDoc = {
        ...sampleDocument,
        id: 'doc2',
        embedding: [0.1, 0.2] // Different dimensions
      };
      expect(() => store.addDocument(invalidDoc)).toThrow('dimensions');
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
    });

    test('should find most similar document', () => {
      const result = store.findMostSimilar([0.1, 0.2, 0.3]);
      expect(result).toBeDefined();
      expect(result?.document.id).toBe('doc1');
    });

    test('should filter by metadata', () => {
      const results = store.search([0.1, 0.2, 0.3], {
        metadataFilter: { category: 'greeting' }
      });
      expect(results.length).toBe(2);
      expect(results[0].document.metadata?.category).toBe('greeting');
    });

    test('should respect search limit', () => {
      const results = store.search([0.1, 0.2, 0.3], { limit: 1 });
      expect(results.length).toBe(1);
    });

    test('should respect similarity threshold', () => {
      const results = store.search([0.1, 0.2, 0.3], { threshold: 0.99 });
      expect(results.length).toBe(0);
    });
  });

  describe('Similarity Methods', () => {
    const vecA = [1, 0, 0];
    const vecB = [0, 1, 0];

    test('should use cosine similarity by default', () => {
      store.addDocument({ ...sampleDocument, embedding: vecA });
      const result = store.search(vecB);
      expect(result[0].similarity).toBeLessThan(1);
    });

    test('should support euclidean similarity', () => {
      store.addDocument({ ...sampleDocument, embedding: vecA });
      const result = store.search(vecB, { similarityMethod: 'euclidean' });
      expect(result[0].similarity).toBeGreaterThan(0);
    });

    test('should support dot product similarity', () => {
      store.addDocument({ ...sampleDocument, embedding: vecA });
      const result = store.search(vecB, { similarityMethod: 'dot' });
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
      expect(() => store.addDocument({ ...sampleDocument, id: '' }))
        .toThrow('Document must have an ID');
    });

    test('should throw on empty embedding', () => {
      expect(() => store.addDocument({ ...sampleDocument, embedding: [] }))
        .toThrow('Document must have a valid embedding');
    });

    test('should throw on invalid embedding values', () => {
      expect(() => store.addDocument({ ...sampleDocument, embedding: [NaN, 0.2, 0.3] }))
        .toThrow('All embedding values must be finite numbers');
    });

    test('should throw on invalid query embedding', () => {
      expect(() => store.search([]))
        .toThrow('Query embedding must be provided and non-empty');
    });
  });
}); 