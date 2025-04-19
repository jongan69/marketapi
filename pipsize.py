import importlib.metadata

total_size = 0
package_sizes = []

for d in importlib.metadata.distributions():
    d_size = 0
    for f in d.files:
        if f.locate().is_file():
            d_size += f.locate().stat().st_size
    total_size += d_size
    package_sizes.append((d_size, d.name))

# Sort packages by size (least to greatest)
package_sizes.sort()

# Display sorted packages
for d_size, name in package_sizes:
    print('{:>12.3f} KiB   {}'.format(d_size/1024, name))

print('-' * 30)
print('{:>12.3f} MB   TOTAL'.format(total_size/(1024*1024)))