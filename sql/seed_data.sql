-- Seed data for cloud_service_pricing
INSERT INTO cloud_service_pricing (service_name, flavor_name, unit, price_per_unit, currency)
VALUES
    ('Cloud Server', 'Small (1vCPU, 2GB RAM)', 'per hour', 5000, 'VND'),
    ('Cloud Server', 'Medium (2vCPU, 4GB RAM)', 'per hour', 10000, 'VND'),
    ('Cloud Server', 'Large (4vCPU, 8GB RAM)', 'per hour', 20000, 'VND'),
    ('SSD Storage', 'SSD Standard', 'per GB per month', 500, 'VND'),
    ('SSD Storage', 'NVMe SSD', 'per GB per month', 1200, 'VND');
