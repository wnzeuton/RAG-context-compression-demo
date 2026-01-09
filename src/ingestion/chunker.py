from pathlib import Path

def chunk_file_by_sections(file_path: Path):
    """
    Parse a document with:
      - Title
      - Type
      - Section blocks separated by '---'
      - Each block starts with a section header line

    Returns:
        List[dict]: chunks with metadata, section, and clean text
    """
    raw_text = file_path.read_text(encoding="utf-8")
    blocks = raw_text.split("\n---\n")

    title = None
    doc_type = None
    chunks = []

    # Track character offsets
    cursor = 0

    for block in blocks:
        block_len = len(block) + len("\n---\n")

        lines = block.splitlines()
        stripped_lines = [l.strip() for l in lines if l.strip()]

        # Metadata lines
        for line in stripped_lines:
            if line.startswith("Title:"):
                title = line.replace("Title:", "").strip()
            elif line.startswith("Type:"):
                doc_type = line.replace("Type:", "").strip()

        # Section blocks
        # Skip blocks that are only metadata
        section_header = None
        content_lines = []

        for line in stripped_lines:
            if not line.startswith(("Title:", "Type:")):
                if section_header is None:
                    section_header = line  # first non-metadata line = section
                else:
                    content_lines.append(line)

        if section_header and content_lines:
            text = " ".join(content_lines).strip()

            # Compute character offsets
            header_pos = raw_text.find(section_header, cursor)
            content_start = header_pos + len(section_header)
            content_end = content_start + len(text)

            chunks.append({
                "title": title,
                "type": doc_type,
                "section": section_header,
                "start_char": content_start,
                "end_char": content_end,
                "text": text
            })

        cursor += block_len

    return chunks


def chunk_documents_from_files(file_paths):
    """
    Chunk multiple documents into section-based chunks.
    """
    all_chunks = []

    for path in file_paths:
        doc_chunks = chunk_file_by_sections(path)
        for chunk_id, c in enumerate(doc_chunks):
            all_chunks.append({
                "doc_name": path.name,
                "chunk_id": chunk_id,
                **c
            })

    return all_chunks