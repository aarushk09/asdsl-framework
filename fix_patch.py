def fix_file(name, out_name):
    with open(name, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace('\\"\\"\\"', '"""')
    with open(out_name, "w", encoding="utf-8") as f:
        f.write(content)

fix_file("patch_kv.py", "patch_kv_fixed.py")
fix_file("patch_kv2.py", "patch_kv2_fixed.py")
