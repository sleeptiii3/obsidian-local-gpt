import { TFile, Vault, MetadataCache } from "obsidian";
import { Document } from "langchain/document";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OllamaEmbeddings } from "./ollama-embeddings";
import { preprocessContent, splitContent } from "./text-processing";

const MAX_DEPTH = 10;
const MAX_SEARCH_RESULTS = 6;
const documentCache = new Map<string, { mtime: number; embedding?: number[] }>();

interface ProcessingContext {
  vault: Vault;
  metadataCache: MetadataCache;
  currentDocumentPath: string;
}

export async function startProcessing(
  file: TFile,
  vault: Vault,
  metadataCache: MetadataCache
): Promise<Map<string, Document>> {
  const context: ProcessingContext = { vault, metadataCache, currentDocumentPath: file.path };
  return processDocumentForRAG(file, context, new Map(), 0, false);
}

async function processDocumentForRAG(
  file: TFile,
  context: ProcessingContext,
  processedDocs: Map<string, Document>,
  depth: number,
  isBacklink: boolean
): Promise<Map<string, Document>> {
  if (depth > MAX_DEPTH || processedDocs.has(file.path)) {
    return processedDocs;
  }

  try {
    const content = await context.vault.cachedRead(file);
    const newDoc = new Document({
      pageContent: content,
      metadata: { source: file.path, basename: file.basename, stat: file.stat, depth, isBacklink }
    });
    processedDocs.set(file.path, newDoc);

    if (!isBacklink) {
      const linkedFiles = getLinkedFiles(content, context.vault, context.metadataCache, file.path);
      for (const linkedFile of linkedFiles) {
        processedDocs = await processDocumentForRAG(linkedFile, context, processedDocs, depth + 1, false);
      }

      const backlinkFiles = getBacklinkFiles(file, context, processedDocs);
      for (const backlinkFile of backlinkFiles) {
        processedDocs = await processDocumentForRAG(backlinkFile, context, processedDocs, depth, true);
      }
    }
  } catch (error) {
    console.error(`Error processing document ${file.path}:`, error);
  }

  return processedDocs;
}

export function getLinkedFiles(
  content: string,
  vault: Vault,
  metadataCache: MetadataCache,
  currentFilePath: string
): TFile[] {
  const linkRegex = /\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]/g;
  const matches = content.matchAll(linkRegex);
  
  return Array.from(matches, match => match[1])
    .map(linkText => {
      const linkPath = metadataCache.getFirstLinkpathDest(linkText, currentFilePath);
      return linkPath ? vault.getAbstractFileByPath(linkPath.path) : null;
    })
    .filter((file): file is TFile => file instanceof TFile && file.extension === 'md');
}

function getBacklinkFiles(
  file: TFile,
  context: ProcessingContext,
  processedDocs: Map<string, Document>
): TFile[] {
  const resolvedLinks = context.metadataCache.resolvedLinks;
  const backlinkPaths = Object.entries(resolvedLinks)
    .filter(([sourcePath, links]) => links[file.path] && !processedDocs.has(sourcePath))
    .map(([sourcePath]) => sourcePath);

  return backlinkPaths
    .map(path => context.vault.getAbstractFileByPath(path))
    .filter((backlinkFile): backlinkFile is TFile => 
      backlinkFile instanceof TFile && backlinkFile.extension === 'md'
    );
}

export async function createVectorStore(
  documents: Document[], 
  ollamaUrl: string, 
  currentDocumentPath: string
): Promise<MemoryVectorStore> {
  const embeddings = new OllamaEmbeddings(ollamaUrl);
  const vectorStore = new MemoryVectorStore(embeddings);
  const uniqueChunks = new Set<string>();

  for (const doc of documents) {
    if (doc.metadata.source !== currentDocumentPath) {
      const content = preprocessContent(doc.pageContent);
      const chunks = splitContent(content);
      for (const chunk of chunks) {
        if (!uniqueChunks.has(chunk)) {
          uniqueChunks.add(chunk);
          const chunkDoc = new Document({ pageContent: chunk, metadata: { ...doc.metadata } });
          
          try {
            const cachedData = documentCache.get(doc.metadata.source);
            if (cachedData?.embedding) {
              await vectorStore.addVectors([cachedData.embedding], [chunkDoc]);
            } else {
              const [embedding] = await embeddings.embedDocuments([chunk]);
              await vectorStore.addVectors([embedding], [chunkDoc]);
              documentCache.set(doc.metadata.source, { mtime: doc.metadata.stat.mtime, embedding });
            }
          } catch (error) {
            console.error(`Error creating embedding for ${doc.metadata.source}:`, error);
          }
        }
      }
    }
  }

  console.log(`Total unique chunks: ${uniqueChunks.size}`);
  return vectorStore;
}

export async function queryVectorStore(
  query: string,
  vectorStore: MemoryVectorStore
): Promise<string> {
  const results = await vectorStore.similaritySearchWithScore(query, MAX_SEARCH_RESULTS);
  console.log("Results:", results);
  
  const groupedResults = results.reduce((acc, [doc, score]) => {
    const basename = doc.metadata.basename || 'Unknown';
    if (!acc[basename]) {
      acc[basename] = [];
    }
    acc[basename].push(doc.pageContent);
    return acc;
  }, {} as Record<string, string[]>);

  return Object.entries(groupedResults)
    .map(([basename, contents]) => `${basename}\n${contents.join('\n\n')}`)
    .join('\n\n')
    .trim();
}
