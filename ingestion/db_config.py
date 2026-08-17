import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

load_dotenv()


def get_engine():
    host     = os.getenv("DB_HOST", "localhost")
    port     = os.getenv("DB_PORT", "5432")
    db       = os.getenv("DB_NAME", "tanger_med")
    user     = os.getenv("DB_USER", "postgres")
    password = quote_plus(os.getenv("DB_PASSWORD", ""))

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

    engine = create_engine(
        url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True
    )
    return engine


def test_connection():
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"Connexion PostgreSQL réussie !")
            return True
    except Exception as e:
        print(f" Erreur de connexion : {e}")
        print("   Vérifie tes credentials dans .env")
        return False


if __name__ == "__main__":
    test_connection()