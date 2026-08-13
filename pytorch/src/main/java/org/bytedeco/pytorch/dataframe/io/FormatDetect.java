package org.bytedeco.pytorch.dataframe.io;

import org.bytedeco.pytorch.dataframe.DataFrame;

import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/**
 * Extension-based format detection for {@link DataFrame#read(String)}.
 *
 * <p>Supported extensions grow with the I/O stack:
 * {@code .csv .tsv .json .jsonl .ndjson .parquet .arrow .feather .ipc
 * .pkl .pickle .xlsx .xls .h5 .hdf5 .hdf .avro .orc .npz .npy
 * .safetensors .gguf .lance}.
 */
public final class FormatDetect {
    private FormatDetect() {}

    public enum Format {
        CSV, TSV, JSON, JSONL, PARQUET, ARROW, FEATHER, PICKLE,
        EXCEL, HDF5, AVRO, ORC, NPZ, NPY, SAFETENSORS, GGUF, LANCE, TOML, BIN, IMDB, 
        LMDB, IMAGEFOLDER, SOUNDFOLDER, WEBDATASET, TEXT,
        PDF, DOCUMENT, HTML, MARKDOWN, XML, 
        SHAPEFILE, RASTER, GEOTIFF,
        VOXEL, MESH, POINTCLOUD,
        UNKNOWN
    }

    public static Format detect(String path) {
        if (path == null || path.isEmpty()) return Format.UNKNOWN;
        String name = path;
        int slash = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
        if (slash >= 0 && slash + 1 < path.length()) name = path.substring(slash + 1);
        String lower = name.toLowerCase(Locale.ROOT);

        if (lower.endsWith(".safetensors")) return Format.SAFETENSORS;
        if (lower.endsWith(".jsonl") || lower.endsWith(".ndjson")) return Format.JSONL;
        if (lower.endsWith(".lance")) return Format.LANCE;

        // Directory heuristics for Lance datasets (no file extension required)
        try {
            Path p = Path.of(path);
            if (Files.isDirectory(p)) {
                if (Files.isRegularFile(p.resolve("_manifest.json"))
                    || Files.isDirectory(p.resolve("_versions"))
                    || (Files.isDirectory(p.resolve("data"))
                        && (Files.isDirectory(p.resolve("_versions"))
                            || Files.isDirectory(p.resolve("indices"))
                            || Files.isDirectory(p.resolve("vectors"))))) {
                    return Format.LANCE;
                }
            }
        } catch (Exception ignored) {
            // fall through to extension switch
        }

        int dot = lower.lastIndexOf('.');
        if (dot < 0 || dot == lower.length() - 1) return Format.UNKNOWN;
        String ext = lower.substring(dot + 1);
        switch (ext) {
            case "csv": return Format.CSV;
            case "tsv": return Format.TSV;
            case "json": return Format.JSON;
            case "parquet": case "pq": return Format.PARQUET;
            case "arrow": case "ipc": return Format.ARROW;
            case "feather": return Format.FEATHER;
            case "pkl": case "pickle": return Format.PICKLE;
            case "xlsx": case "xls": case "xlsm": return Format.EXCEL;
            case "h5": case "hdf5": case "hdf": return Format.HDF5;
            case "avro": return Format.AVRO;
            case "orc": return Format.ORC;
            case "npz": return Format.NPZ;
            case "npy": return Format.NPY;
            case "gguf": return Format.GGUF;
            case "lance": return Format.LANCE;
            case "toml": return Format.TOML;
            case "bin": return Format.BIN;
            case "imdb": return Format.IMDB;
            case "lmdb": case "mdb": return Format.LMDB;
            case "imagefolder": case "images": return Format.IMAGEFOLDER;
            case "soundfolder": case "audiofolder": case "audio": return Format.SOUNDFOLDER;
            case "webdataset": case "wds": case "tar": return Format.WEBDATASET;
            case "txt": case "text": return Format.TEXT;
            case "pdf": return Format.PDF;
            case "doc": case "docx": case "html": case "htm": 
            case "markdown": case "md": case "xml": return Format.DOCUMENT;
            case "shp": return Format.SHAPEFILE;
            case "tif": case "tiff": case "dem": case "bil": return Format.RASTER;
            case "geotiff": case "gtiff": return Format.GEOTIFF;
            case "binvox": case "vox": case "raw3d": return Format.VOXEL;
            case "obj": case "ply": case "stl": case "off": case "mesh": return Format.MESH;
            case "pcd": case "xyz": case "las": case "laz": return Format.POINTCLOUD;
            default: return Format.UNKNOWN;
        }
    }

    /**
     * Detect LMDB by directory structure or file content.
     */
    private static boolean isLmdbDirectory(Path p) {
        if (!Files.isDirectory(p)) return false;
        return Files.exists(p.resolve("data.mdb")) 
            || Files.exists(p.resolve("train.mdb"))
            || Files.exists(p.resolve("test.mdb"))
            || Files.exists(p.resolve("lock.mdb"));
    }

    /**
     * Detect ImageFolder by checking for image subdirectories.
     */
    private static boolean isImageFolderDirectory(Path p) {
        if (!Files.isDirectory(p)) return false;
        // ImageFolder has subdirectories with class names containing images
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(p)) {
            for (Path entry : stream) {
                if (Files.isDirectory(entry)) {
                    // Check if directory contains image files
                    try (DirectoryStream<Path> images = Files.newDirectoryStream(entry, "*.{jpg,JPG,png,PNG,gif,GIF,bmp,BMP,webp,WEBP}")) {
                        if (images.iterator().hasNext()) {
                            return true;
                        }
                    }
                }
            }
        } catch (Exception e) {}
        return false;
    }

    /**
     * Detect SoundFolder by checking for audio subdirectories.
     */
    private static boolean isSoundFolderDirectory(Path p) {
        if (!Files.isDirectory(p)) return false;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(p)) {
            for (Path entry : stream) {
                if (Files.isDirectory(entry)) {
                    // Check if directory contains audio files
                    try (DirectoryStream<Path> audio = Files.newDirectoryStream(entry, "*.{wav,mp3,flac,ogg,m4a,aac}")) {
                        if (audio.iterator().hasNext()) {
                            return true;
                        }
                    }
                }
            }
        } catch (Exception e) {}
        return false;
    }

    public static Format detect(Path path) {
        return path == null ? Format.UNKNOWN : detect(path.toString());
    }

    /**
     * Detect format by extension, falling back to magic-byte sniff
     * ({@link SchemaInfer#sniff(String)}) when the extension is unknown.
     * Use this for robust multi-format loading.
     */
    public static Format detectRobust(String path) {
        Format fmt = detect(path);
        if (fmt != Format.UNKNOWN) return fmt;
        
        // Try to detect by directory structure
        try {
            Path p = Path.of(path);
            if (Files.isDirectory(p)) {
                if (isLmdbDirectory(p)) {
                    return Format.LMDB;
                }
                if (isImageFolderDirectory(p)) {
                    return Format.IMAGEFOLDER;
                }
                if (isSoundFolderDirectory(p)) {
                    return Format.SOUNDFOLDER;
                }
            }
        } catch (Exception ignored) {}
        
        return SchemaInfer.sniff(path);
    }

    /**
     * Load a DataFrame by file extension (with magic-byte fallback).
     * HDF5 auto-read uses default key {@code /df}.
     */
    public static DataFrame read(String path) throws Exception {
        Format fmt = detectRobust(path);
        switch (fmt) {
            case CSV:
                return DataFrame.readCsv(path);
            case TSV:
                return DataFrame.readTsv(path);
            case JSON:
                return DataFrame.readJson(path);
            case JSONL:
                return DataFrame.readJsonl(path);
            case PARQUET:
                return DataFrame.readParquet(path);
            case ARROW:
            case FEATHER:
                return DataFrame.readArrow(path);
            case PICKLE:
                return DataFrame.readPickle(path);
            case NPZ:
                return DataFrame.readNpz(path);
            case NPY:
                return DataFrame.readNpy(path);
            case SAFETENSORS:
                return DataFrame.readSafetensors(path);
            case GGUF:
                return DataFrame.readGguf(path);
            case EXCEL:
                return DataFrame.readExcel(path);
            case HDF5:
                return DataFrame.readHdf(path, "/df");
            case AVRO:
                return DataFrame.readAvro(path);
            case ORC:
                return DataFrame.readOrc(path);
            case LANCE:
                return DataFrame.readLance(path);
            case TOML:
                return TomlReader.read(path);
            case BIN:
                return BinReader.read(path);
            case LMDB:
                return org.bytedeco.pytorch.dataframe.io.lmdb.LmdbReader.read(path);
            case IMAGEFOLDER:
                return org.bytedeco.pytorch.dataframe.io.config.ImageFolderReader.read(path);
            case SOUNDFOLDER:
                return org.bytedeco.pytorch.dataframe.io.config.SoundFolderReader.read(path);
            case WEBDATASET:
                return org.bytedeco.pytorch.dataframe.io.config.WebDatasetReader.read(path);
            case TEXT:
                return org.bytedeco.pytorch.dataframe.io.text.TextCorpusReader.read(path);
            case PDF:
                return org.bytedeco.pytorch.dataframe.io.config.PdfReader.read(path);
            case DOCUMENT:
                return org.bytedeco.pytorch.dataframe.io.config.DocumentReader.read(path);
            case SHAPEFILE:
                return org.bytedeco.pytorch.dataframe.io.config.ShapefileReader.read(path);
            case RASTER:
            case GEOTIFF:
                return org.bytedeco.pytorch.dataframe.io.config.RasterReader.read(path);
            case VOXEL:
                return org.bytedeco.pytorch.data.multimodal.VoxelData.fromFile(path).toDataFrame();
            case MESH:
                return org.bytedeco.pytorch.data.multimodal.MeshData.fromFile(path).toDataFrame();
            case POINTCLOUD:
                return org.bytedeco.pytorch.dataframe.dtype.PointCloudData.fromFile(path).toDataFrame();
            default:
                throw new IllegalArgumentException(
                    "Cannot auto-detect DataFrame format for path: " + path
                        + " (supported: csv,tsv,json,jsonl,parquet,arrow,feather,ipc,"
                        + "pkl,xlsx,xls,h5,hdf5,avro,orc,npz,npy,safetensors,gguf,lance,toml,bin;"
                        + " also magic-byte sniff for PAR1/ARROW1/NUMPY/ORC/HDF/Avro/JSON)");
        }
    }
}
