const MAX_CHUNK_SIZE = 1000;

export function preprocessContent(content: string): string {
  return content
    .replace(/^---\n[\s\S]*?\n---\n/, '')
    .replace(/```[\s\S]*?```/g, '')
    .replace(/^#+\s*[^\n]+\n(?=#+|\s*$)/gm, '')
    .replace(/\n{3,}/g, '\n\n');
}

export function splitContent(content: string): string[] {
  const sections = content.split(/^---$/m).flatMap(section => splitByHeaders(section.trim()));
  return sections.flatMap(section => 
    section.length <= MAX_CHUNK_SIZE ? [section] : splitLargeSection(section)
  ).filter(chunk => chunk.length > 0);
}

function splitByHeaders(text: string): string[] {
  const headerRegex = /^(#{1,6})\s+(.+)$/gm;
  const sections: string[] = [];
  let lastIndex = 0;
  let lastHeader = '';
  let match;

  while ((match = headerRegex.exec(text)) !== null) {
    if (lastIndex < match.index) {
      sections.push(lastHeader + text.slice(lastIndex, match.index).trim());
    }
    lastHeader = match[0] + '\n';
    lastIndex = headerRegex.lastIndex;
  }

  if (lastIndex < text.length) {
    sections.push(lastHeader + text.slice(lastIndex).trim());
  }

  return sections.filter(section => section.length > 0);
}

function splitLargeSection(section: string): string[] {
  return splitByDoubleLine(section).flatMap(chunk => 
    chunk.length <= MAX_CHUNK_SIZE ? [chunk] : splitTextRecursively(chunk)
  );
}

function splitByDoubleLine(text: string): string[] {
  return text.split(/\n\s*\n/).filter(chunk => chunk.trim().length > 0);
}

function splitTextRecursively(text: string): string[] {
  if (text.length <= MAX_CHUNK_SIZE) {
    return [text];
  }

  const splitByLine = text.split('\n');
  if (splitByLine.length > 1) {
    const midIndex = Math.floor(splitByLine.length / 2);
    const firstHalf = splitByLine.slice(0, midIndex).join('\n');
    const secondHalf = splitByLine.slice(midIndex).join('\n');
    return [...splitTextRecursively(firstHalf), ...splitTextRecursively(secondHalf)];
  }

  const splitByPeriod = text.split('.');
  if (splitByPeriod.length > 1) {
    const midIndex = Math.floor(splitByPeriod.length / 2);
    const firstHalf = splitByPeriod.slice(0, midIndex).join('.');
    const secondHalf = splitByPeriod.slice(midIndex).join('.');
    return [...splitTextRecursively(firstHalf), ...splitTextRecursively(secondHalf)];
  }

  return [text];
}