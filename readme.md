# NeuraDB

**A lightweight, zero-dependency in-memory vector database for blazing fast similarity search with OpenAI integration.**

![npm](https://img.shields.io/npm/v/neuradb)
![license](https://img.shields.io/npm/l/neuradb)
![typescript](https://img.shields.io/badge/TypeScript-Ready-blue)

NeuraDB enables high-performance vector similarity search in TypeScript/JavaScript environments, supporting multiple similarity methods, metadata filtering, memory-efficient storage, and automatic OpenAI embeddings.

## ✨ Features

- 🧠 Fast in-memory vector search
- 🔢 Cosine, Euclidean, Dot Product similarity
- 🏷️ Metadata filtering
- 📦 Zero dependencies (OpenAI optional)
- ⚡ Memory-efficient
- 🧪 TypeScript-first API
- 🤖 Automatic OpenAI embeddings
- 📈 Batch processing with progress tracking
- 🔄 Automatic embedding dimension validation

## 📦 Installation

```bash
# Using npm
npm install neuradb

# Using Yarn
yarn add neuradb
```

## 🧰 Usage

### Basic Usage

```typescript
import { NeuraDB } from "neuradb";

const db = new NeuraDB();

// Add a document with pre-computed embedding
db.addDocument({
  id: "doc1",
  content: "Hello world",
  embedding: [0.1, 0.2, 0.3],
  metadata: { category: "greeting" },
});

// Search for similar documents
const results = db.search([0.1, 0.2, 0.3], {
  limit: 5,
  threshold: 0.7,
  similarityMethod: "cosine",
});

console.log(results);
```

### With OpenAI Integration

```typescript
import { NeuraDB } from "neuradb";
import OpenAI from "openai";

const openai = new OpenAI({ apiKey: "your-api-key" });
const db = new NeuraDB({
  openai,
  embeddingModel: "text-embedding-3-small",
  defaultBatchSize: 100,
  batchDelay: 1000,
});

// Add document with automatic embedding generation
await db.addDocument(
  {
    id: "doc1",
    content: "Hello world",
    metadata: { category: "greeting" },
  },
  { createEmbedding: true }
);

// Search with text query (automatic embedding)
const results = await db.search("Hello there", {
  limit: 5,
  threshold: 0.7,
  similarityMethod: "cosine",
  metadataFilter: { category: "greeting" },
});

// Batch add documents with progress tracking
await db.addDocuments(
  [
    { id: "1", content: "Text 1" },
    { id: "2", content: "Text 2" },
  ],
  {
    createEmbedding: true,
    batchSize: 50,
    batchDelay: 500,
    onProgress: (processed, total) => console.log(`${processed}/${total}`),
  }
);
```

## 🔍 API Overview

### Core Methods

| Method                                                                              | Description                                                  |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `addDocument(document: VectorDocument, options?: { createEmbedding?: boolean })`    | Add a single document with optional embedding generation     |
| `addDocuments(documents: VectorDocument[], options?: AddDocumentsOptions)`          | Batch-add multiple documents with progress tracking          |
| `search(query: number[] \| string, options?: SearchOptions)`                        | Find documents most similar to the query (embedding or text) |
| `findMostSimilar(query: number[] \| string, similarityMethod?: SimilarityMethod)`   | Return the single most similar document                      |
| `updateDocument(document: VectorDocument, options?: { createEmbedding?: boolean })` | Update an existing document by ID                            |
| `getDocument(id: string)`                                                           | Fetch a document by its ID                                   |
| `getDocumentsByMetadata(filter: Record<string, any>)`                               | Retrieve documents matching metadata criteria                |
| `clear()`                                                                           | Clear the entire store                                       |

### Configuration Methods

| Method                                   | Description                                     |
| ---------------------------------------- | ----------------------------------------------- |
| `setEmbeddingModel(model: string)`       | Set the OpenAI embedding model                  |
| `setDefaultBatchSize(batchSize: number)` | Set default batch size for embedding operations |
| `setDefaultBatchDelay(delay: number)`    | Set delay between batches in milliseconds       |
| `getStats()`                             | Get store statistics                            |

## 🧠 Similarity Methods

| Method      | Description                           |
| ----------- | ------------------------------------- |
| `cosine`    | Measures cosine angle between vectors |
| `euclidean` | Measures Euclidean distance           |
| `dot`       | Dot product of two vectors            |

## 📊 Stats

```typescript
const stats = db.getStats();
console.log(stats);
// {
//   documentCount: 100,
//   embeddingDimensions: 384,
//   estimatedMemoryUsage: 250000
// }
```

## 📁 Types

```typescript
/**
 * Represents a document with its vector embedding and metadata
 */
interface VectorDocument {
  /** Unique identifier for the document */
  id: string;

  /** The text content of the document */
  content: string;

  /** Vector embedding representation of the document */
  embedding: number[];

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
interface SearchResult {
  /** The document that matched the search */
  document: VectorDocument;

  /** Similarity score between 0 and 1 (1 being most similar) */
  similarity: number;
}

/**
 * Supported similarity calculation methods
 */
type SimilarityMethod = "cosine" | "euclidean" | "dot";

/**
 * Configuration options for vector search
 */
interface SearchOptions {
  /** Maximum number of results to return */
  limit?: number;

  /** Minimum similarity threshold (0-1) */
  threshold?: number;

  /** Similarity calculation method to use */
  similarityMethod?: SimilarityMethod;

  /** Metadata filters to apply */
  metadataFilter?: Record<string, any>;
}

/**
 * Options for adding multiple documents
 */
interface AddDocumentsOptions {
  /** Whether to generate embeddings for documents without them */
  createEmbedding?: boolean;

  /** Number of documents to process in each batch */
  batchSize?: number;

  /** Delay between batches in milliseconds */
  batchDelay?: number;

  /** Progress callback function */
  onProgress?: (processed: number, total: number) => void;
}

/**
 * Statistics about the vector store
 */
interface VectorStoreStats {
  /** Total number of documents stored */
  documentCount: number;

  /** Dimensions of the vector embeddings */
  embeddingDimensions: number | null;

  /** Memory usage estimation in bytes */
  estimatedMemoryUsage: number;
}
```

## 📃 License

MIT © Haider Nakara

## 💬 Acknowledgements

Inspired by the simplicity of ChromaDB, Pinecone, and Faiss — with a developer-friendly twist.

## 🔗 Contributing

PRs, feature requests, and issues are welcome. Let's build better vector tooling together!
