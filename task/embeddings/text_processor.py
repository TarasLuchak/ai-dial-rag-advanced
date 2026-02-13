from enum import StrEnum

import psycopg2
from psycopg2.extras import RealDictCursor

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config["host"],
            port=self.db_config["port"],
            database=self.db_config["database"],
            user=self.db_config["user"],
            password=self.db_config["password"],
            cursor_factory=RealDictCursor,
        )

    #TODO:
    # provide method `process_text_file` that will:
    #   - apply file name, chunk size, overlap, dimensions and bool of the table should be truncated
    #   - truncate table with vectors if needed
    #   - load content from file and generate chunks (in `utils.text` present `chunk_text` that will help do that)
    #   - generate embeddings from chunks
    #   - save (insert) embeddings and chunks to DB
    #       hint 1: embeddings should be saved as string list
    #       hint 2: embeddings string list should be casted to vector ({embeddings}::vector)




    #TODO:
    # provide method `search` that will:
    #   - apply search mode, user request, top k for search, min score threshold and dimensions
    #   - generate embeddings from user request
    #   - search in DB relevant context
    #     hint 1: to search it in DB you need to create just regular select query
    #     hint 2: Euclidean distance `<->`, Cosine distance `<=>`
    #     hint 3: You need to extract `text` from `vectors` table
    #     hint 4: You need to filter distance in WHERE clause
    #     hint 5: To get top k use `limit`

    def _truncate_table(self, cursor) -> None:
        """Remove all rows from the vectors table."""
        cursor.execute("TRUNCATE TABLE vectors")

    def _save_chunk(
            self,
            cursor,
            document_name: str,
            text: str,
            embedding: list[float],
    ) -> None:
        """
        Persist a single text chunk and its embedding into the database.

        Embedding is stored as a pgvector, cast from its textual representation.
        """
        if not embedding:
            return

        embedding_str = ",".join(str(value) for value in embedding)
        vector_literal = f"[{embedding_str}]"

        cursor.execute(
            """
            INSERT INTO vectors (document_name, text, embedding)
            VALUES (%s, %s, %s::vector)
            """,
            (document_name, text, vector_literal),
        )

    def process_text_file(
            self,
            file_path: str,
            document_name: str | None = None,
            chunk_size: int = 150,
            overlap: int = 40,
            dimensions: int = 1536,
            truncate_table: bool = True,
    ) -> None:
        """
        Load a text file, split it into chunks, embed them and store in the DB.
        """
        if document_name is None:
            document_name = file_path.split("/")[-1].split("\\")[-1]

        with self._get_connection() as conn, conn.cursor() as cursor:
            if truncate_table:
                self._truncate_table(cursor)

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)

            if not chunks:
                return

            embeddings = self.embeddings_client.get_embeddings(
                chunks,
                dimensions=dimensions,
            )

            for idx, chunk in enumerate(chunks):
                embedding = embeddings.get(idx)
                if embedding is None:
                    continue
                self._save_chunk(cursor, document_name, chunk, embedding)

            conn.commit()

    def search(
            self,
            search_mode: SearchMode,
            user_request: str,
            top_k: int = 5,
            min_score: float = 0.5,
            dimensions: int = 1536,
    ) -> list[dict]:
        """
        Perform semantic search over stored vectors and return the most relevant chunks.
        """
        if not user_request:
            return []

        query_embeddings = self.embeddings_client.get_embeddings(
            [user_request],
            dimensions=dimensions,
        )

        query_vector = query_embeddings.get(0)
        if not query_vector:
            return []

        embedding_str = ",".join(str(value) for value in query_vector)
        vector_literal = f"[{embedding_str}]"

        if search_mode == SearchMode.COSINE_DISTANCE:
            operator = "<=>"
        else:
            operator = "<->"

        # Smaller distance means higher similarity; we treat `min_score` as
        # the maximum acceptable distance threshold.
        with self._get_connection() as conn, conn.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    document_name,
                    text,
                    (embedding {operator} %s::vector) AS distance
                FROM vectors
                WHERE (embedding {operator} %s::vector) <= %s
                ORDER BY distance ASC
                LIMIT %s
                """,
                (vector_literal, vector_literal, min_score, top_k),
            )

            rows = cursor.fetchall()

        return list(rows)
