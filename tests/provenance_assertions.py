def assert_manifest_provenance(manifest):
    provenance = manifest["provenance"]
    assert provenance["generated_at_utc"]
    assert isinstance(provenance["command"], list)
    assert provenance["cwd"]
    assert provenance["python"]["version"]
    assert provenance["python"]["executable"]
    assert provenance["python"]["platform"]

    packages = provenance["packages"]
    for package_name in ("efficient-kan", "numpy", "pandas", "scikit-learn", "torch"):
        assert package_name in packages

    repositories = provenance["repositories"]
    assert repositories["kan-d-iv-late"]["commit"]
    assert isinstance(repositories["kan-d-iv-late"]["dirty"], bool)
    assert "efficient-kan" in repositories
