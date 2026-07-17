-- Seed data for cloud_competitor_mapping
INSERT INTO cloud_competitor_mapping (competitor_name, competitor_service_name, local_service_name, feature_comparison, migration_difficulty)
VALUES
    ('AWS', 'S3', 'Object Storage', 'Same functionality, different API structure', 'Easy'),
    ('AWS', 'RDS', 'Managed Database', 'Both fully managed, similar pricing models', 'Medium'),
    ('AWS', 'EC2', 'Cloud Server', 'Similar VM offerings, different management interface', 'Medium'),
    ('Azure', 'Blob Storage', 'Object Storage', 'Equivalent object storage with Azure integration', 'Easy'),
    ('GCP', 'Cloud SQL', 'Managed Database', 'Comparable managed database service', 'Medium');

-- Seed data for cloud_service_pricing
INSERT INTO cloud_service_pricing (service_name, flavor_name, unit, price_per_unit, currency)
VALUES
    ('Cloud Server', 'Small (1vCPU, 2GB RAM)', 'per hour', 5000, 'VND'),
    ('Cloud Server', 'Medium (2vCPU, 4GB RAM)', 'per hour', 10000, 'VND'),
    ('Cloud Server', 'Large (4vCPU, 8GB RAM)', 'per hour', 20000, 'VND'),
    ('SSD Storage', 'SSD Standard', 'per GB per month', 500, 'VND'),
    ('SSD Storage', 'NVMe SSD', 'per GB per month', 1200, 'VND');
