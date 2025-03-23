git ls-files | ForEach-Object {
    $file = $_
    if (git cat-file -e HEAD:"$file") {
        if (git cat-file -p HEAD:"$file" | Select-String -Pattern '^oid sha256') {
            Remove-Item -Force $file
        }
    }
}