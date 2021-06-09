
SOURCE_PATH=$1
BASE_DST=${3:-/scr/scr-with-most-space}
DST_PATH=$BASE_DST/$2
TAR_NAME=$(basename $SOURCE_PATH)

if [ ! -d "$DST_PATH" ]; then
  mkdir -p $DST_PATH
fi

if [ ! -f "$DST_PATH/$TAR_NAME" ]; then
  echo "Copying files to $DST_PATH"
  cp $SOURCE_PATH $DST_PATH
  tar -C $DST_PATH -xzf $DST_PATH/$TAR_NAME
fi

