package org.bytedeco.pytorch.dataframe.io.config;

public  class ImageFolderOptions {
    private boolean recursive = true;
    private boolean includePath = false;
    private boolean includeSize = false;
    private boolean includeModifiedTime = false;
    private int maxImagesPerClass = 0;  // 0 = no limit

    public static ImageFolderOptions defaults() {
        return new ImageFolderOptions();
    }

    public ImageFolderOptions recursive(boolean v) { this.recursive = v; return this; }
    public ImageFolderOptions includePath(boolean v) { this.includePath = v; return this; }
    public ImageFolderOptions includeSize(boolean v) { this.includeSize = v; return this; }
    public ImageFolderOptions includeModifiedTime(boolean v) { this.includeModifiedTime = v; return this; }
    public ImageFolderOptions maxImagesPerClass(int v) { this.maxImagesPerClass = v; return this; }

    public boolean recursive() { return recursive; }
    public boolean includePath() { return includePath; }
    public boolean includeSize() { return includeSize; }
    public boolean includeModifiedTime() { return includeModifiedTime; }
    public int maxImagesPerClass() { return maxImagesPerClass; }
}