for file in *.blk; do
    echo "Contents of $file:"
    od -t dI -An "$file"
done

