from config.settings import TG_HOST, TG_TOKEN
from src.tg.local_store import LocalGraphStore


def get_client():
    """Get a graph client - TigerGraph if configured, else local store."""
    if TG_HOST and "YOUR_SAVANNA_HOST" not in TG_HOST:
        try:
            import pyTigerGraph as tg
            conn = tg.TigerGraphConnection(host=TG_HOST, token=TG_TOKEN)
            print(f"Connected to TigerGraph: {TG_HOST}")
            return conn
        except Exception as e:
            print(f"TigerGraph connection failed: {e}, falling back to local store")

    print("Using local in-memory graph store")
    store = LocalGraphStore()
    from config.settings import CORPUS_PATH
    store.load_corpus(CORPUS_PATH)
    return store
