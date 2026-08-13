package org.bytedeco.pytorch.dataframe.io.config;

public  class SoundFolderOptions {
    private boolean recursive = true;
    private boolean includePath = false;
    private boolean includeSize = false;
    private boolean includeExtension = false;
    private boolean includeModifiedTime = false;
    private boolean includeName = false;

    public static SoundFolderOptions defaults() {
        return new SoundFolderOptions();
    }

    public SoundFolderOptions recursive(boolean v) { this.recursive = v; return this; }
    public SoundFolderOptions includePath(boolean v) { this.includePath = v; return this; }
    public SoundFolderOptions includeSize(boolean v) { this.includeSize = v; return this; }
    public SoundFolderOptions includeExtension(boolean v) { this.includeExtension = v; return this; }
    public SoundFolderOptions includeModifiedTime(boolean v) { this.includeModifiedTime = v; return this; }
    public SoundFolderOptions includeName(boolean v) { this.includeName = v; return this; }

    public boolean recursive() { return recursive; }
    public boolean includePath() { return includePath; }
    public boolean includeSize() { return includeSize; }
    public boolean includeExtension() { return includeExtension; }
    public boolean includeModifiedTime() { return includeModifiedTime; }
    public boolean includeName() { return includeName; }
}