# NeuraDB

**A lightweight, zero-dependency in-memory vector database for blazing fast similarity search.**

![npm](https://img.shields.io/npm/v/neuradb)
![license](https://img.shields.io/npm/l/neuradb)
![typescript](https://img.shields.io/badge/TypeScript-Ready-blue)

NeuraDB enables high-performance vector similarity search in TypeScript/JavaScript environments, supporting multiple similarity methods, metadata filtering, and memory-efficient storage.

## ✨ Features

- 🧠 Fast in-memory vector search
- 🔢 Cosine, Euclidean, Dot Product similarity
- 🏷️ Metadata filtering
- 📦 Zero dependencies
- ⚡ Memory-efficient
- 🧪 TypeScript-first API

## 📦 Installation

```bash
# Using npm
npm install neuradb

# Using Yarn
yarn add neuradb
```

## 🧰 Usage

```typescript
import { NeuraDB } from "neuradb";

const db = new NeuraDB();

// Add a document
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

## 🔍 API Overview

### Core Methods

| Method                                                                           | Description                                        |
| -------------------------------------------------------------------------------- | -------------------------------------------------- |
| `addDocument(document: VectorDocument)`                                          | Add a single document with an embedding            |
| `addDocuments(documents: VectorDocument[])`                                      | Batch-add multiple documents                       |
| `search(queryEmbedding: number[], options?: SearchOptions)`                      | Find documents most similar to the query embedding |
| `findMostSimilar(queryEmbedding: number[], similarityMethod?: SimilarityMethod)` | Return the single most similar document            |
| `updateDocument(document: VectorDocument)`                                       | Update an existing document by ID                  |
| `getDocument(id: string)`                                                        | Fetch a document by its ID                         |
| `getDocumentsByMetadata(filter: Record<string, any>)`                            | Retrieve documents matching metadata criteria      |
| `clear()`                                                                        | Clear the entire store                             |

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
type SimilarityMethod = "cosine" | "euclidean" | "dot";

interface VectorDocument {
  id: string;
  content: string;
  embedding: number[];
  metadata?: Record<string, any>;
  createdAt?: Date;
  updatedAt?: Date;
}

interface SearchOptions {
  limit?: number;
  threshold?: number;
  similarityMethod?: SimilarityMethod;
  metadataFilter?: Record<string, any>;
}

interface SearchResult {
  document: VectorDocument;
  similarity: number;
}

interface VectorStoreStats {
  documentCount: number;
  embeddingDimensions: number | null;
  estimatedMemoryUsage: number;
}
```

## 📃 License

MIT © Haider Nakara

## 💬 Acknowledgements

Inspired by the simplicity of ChromaDB, Pinecone, and Faiss — with a developer-friendly twist.

## 🔗 Contributing

PRs, feature requests, and issues are welcome. Let's build better vector tooling together!

---

Would you like me to also:

- Prepare the NPM `package.json` boilerplate?
- Scaffold a demo with example embeddings and queries?
- Generate types and JSDocs for publishing?

Let me know how far you want to go with this!
