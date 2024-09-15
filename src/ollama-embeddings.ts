import { Embeddings } from "langchain/embeddings/base";

export class OllamaEmbeddings extends Embeddings {
  private ollamaUrl: string;
  private model: string;

  constructor(ollamaUrl: string, model: string = "bge-m3") {
    super();
    this.ollamaUrl = ollamaUrl;
    this.model = model;
  }

  async embedDocuments(documents: string[]): Promise<number[][]> {
    return this.embed(documents);
  }

  async embedQuery(text: string): Promise<number[]> {
    const result = await this.embed([text]);
    return result[0];
  }

  async embed(texts: string[]): Promise<number[][]> {
    const embeddings = await Promise.all(texts.map(text => this.embedSingle(text)));
    return embeddings;
  }

  private async embedSingle(text: string): Promise<number[]> {
    const response = await fetch(`${this.ollamaUrl}/api/embeddings`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: this.model,
        prompt: text,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    return result.embedding;
  }
}
