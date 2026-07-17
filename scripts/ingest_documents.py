"""
Ingestion script for sample document chunks to Supabase.
Creates 10 sample documents with embeddings for testing RAG queries.
"""

from app.db.supabase_client import supabase
from app.tools.embedding_tool import embed_text


SAMPLE_DOCS = [
    {
        "service_name": "Object Storage",
        "category": "SLA",
        "content": "Object Storage guarantees 99.9% monthly uptime SLA. If this is not met, the customer receives a 10% service fee credit for that month under the compensation policy.",
        "metadata": {"source": "sample_data", "type": "sla_commitment"}
    },
    {
        "service_name": "Cloud Server",
        "category": "Security",
        "content": "The Cloud Server system holds ISO 27001 certification for information security management and complies with the PCI-DSS standard for payment card data processing.",
        "metadata": {"source": "sample_data", "type": "certification"}
    },
    {
        "service_name": "Managed Database",
        "category": "Compliance",
        "content": "The Managed Database service complies with Decree 13/2023/ND-CP on personal data protection, ensuring customer data is stored in domestic data centers.",
        "metadata": {"source": "sample_data", "type": "legal"}
    },
    {
        "service_name": "Object Storage",
        "category": "Feature",
        "content": "Object Storage provides unlimited scalability with multi-region replication. It supports lifecycle policies, encryption at rest and in transit, and versioning of objects. Perfect for backup and archival use cases.",
        "metadata": {"source": "sample_data", "type": "feature"}
    },
    {
        "service_name": "Cloud Server",
        "category": "Feature",
        "content": "Cloud Server instances are available in multiple configurations from 1vCPU to 96vCPU with flexible memory options. Auto-scaling groups allow dynamic resource allocation based on demand.",
        "metadata": {"source": "sample_data", "type": "feature"}
    },
    {
        "service_name": "Managed Database",
        "category": "Security",
        "content": "Database backups are encrypted and stored in a separate geographic region for disaster recovery. All connections use TLS 1.3 encryption, and database access is controlled via IAM policies.",
        "metadata": {"source": "sample_data", "type": "security"}
    },
    {
        "service_name": "Managed Database",
        "category": "SLA",
        "content": "Managed Database guarantees 99.95% availability with automatic failover to a standby replica. Recovery Point Objective (RPO) is less than 5 minutes, and Recovery Time Objective (RTO) is under 2 minutes.",
        "metadata": {"source": "sample_data", "type": "sla"}
    },
    {
        "service_name": "Cloud Server",
        "category": "Compliance",
        "content": "Cloud Server infrastructure complies with Vietnam Data Center and Cloud Service Security Standard (VDSS). All physical locations are audited annually for compliance and security standards.",
        "metadata": {"source": "sample_data", "type": "legal"}
    },
    {
        "service_name": "Object Storage",
        "category": "Compliance",
        "content": "Object Storage uses HMAC-SHA256 signing and supports server-side encryption with customer-provided keys (SSE-C) or service-managed keys. Audit logging is available for all object operations.",
        "metadata": {"source": "sample_data", "type": "compliance"}
    },
    {
        "service_name": "Network",
        "category": "Feature",
        "content": "Global Load Balancing service distributes incoming traffic across multiple regions and availability zones. Supports advanced routing policies including weighted, latency-based, and geolocation routing.",
        "metadata": {"source": "sample_data", "type": "feature"}
    }
]


def ingest():
    """Ingests sample documents with embeddings into Supabase."""
    for doc in SAMPLE_DOCS:
        embedding = embed_text(doc["content"])
        supabase.table("cloud_documents").insert({
            "service_name": doc["service_name"],
            "category": doc["category"],
            "content": doc["content"],
            "embedding": embedding,
            "metadata": doc["metadata"]
        }).execute()
        print(f"✓ Inserted: {doc['service_name']} - {doc['category']}")


if __name__ == "__main__":
    print("Starting document ingestion...")
    try:
        ingest()
        print("\n✅ All documents ingested successfully!")
    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        print("Note: Ensure Supabase is properly configured and cloud_documents table exists with pgvector support.")
