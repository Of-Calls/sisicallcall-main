import asyncio
import asyncpg

CALL_ID = "1fc3c309-2c9d-4cd4-9378-d9d8ed4b1b3c"

async def main():
    conn = await asyncpg.connect(
        user="sisicallcall",
        password="changeme",
        database="sisicallcall",
        host="127.0.0.1",
        port=5432,
        ssl=False,
    )
    rows = await conn.fetch(
        """
        SELECT turn_index, speaker, text
        FROM transcripts
        WHERE call_id = $1::uuid
        ORDER BY turn_index
        """,
        CALL_ID,
    )
    print("rows:", len(rows))
    for r in rows:
        print(dict(r))
    await conn.close()

asyncio.run(main())