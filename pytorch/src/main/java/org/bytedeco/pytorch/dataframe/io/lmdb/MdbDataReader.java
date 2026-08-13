package org.bytedeco.pytorch.dataframe.io.lmdb;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * Pure-Java LMDB file reader for data.mdb files.
 * Handles the MDB (LMDB data) file format directly without native dependencies.
 */
public class MdbDataReader {

    public static void main(String[] args) throws Exception {
        String path = args.length > 0 ? args[0] : "/home/muller/下载/data.lmdb";
        System.out.println("=== LMDB MDB Data Reader ===");
        System.out.println("File: " + path);
        
        File file = new File(path);
        if (!file.exists()) {
            System.out.println("File not found: " + path);
            return;
        }
        
        System.out.println("File size: " + (file.length() / 1024 / 1024) + " MB");
        System.out.println();
        
        try (FileChannel ch = FileChannel.open(file.toPath(), StandardOpenOption.READ)) {
            readMdbFile(ch, file.length());
        }
    }
    
    static void readMdbFile(FileChannel channel, long fileSize) throws Exception {
        // Try common page sizes
        for (int pageSize : new int[]{4096, 8192, 16384, 2048}) {
            if (fileSize < pageSize * 2) continue;
            
            try {
                analyzeFile(channel, pageSize, fileSize);
                return;
            } catch (Exception e) {
                // Try next page size
            }
        }
        
        System.out.println("Could not determine page size");
    }
    
    static void analyzeFile(FileChannel channel, int pageSize, long fileSize) throws Exception {
        ByteBuffer meta0 = readPage(channel, 0, pageSize);
        ByteBuffer meta1 = readPage(channel, 1, pageSize);
        
        // Check meta page magic
        // In LMDB, meta page contains database state
        // The first bytes tell us if this is a valid meta page
        
        int meta0Type = meta0.getShort(0) & 0xFFFF;
        int meta1Type = meta1.getShort(0) & 0xFFFF;
        
        System.out.println("Meta page 0 type: 0x" + Integer.toHexString(meta0Type));
        System.out.println("Meta page 1 type: 0x" + Integer.toHexString(meta1Type));
        
        // Find which meta page is valid by checking txnid
        long txnid0 = meta0.getLong(32);
        long txnid1 = meta1.getLong(32);
        
        System.out.println("Meta 0 txnid: " + txnid0);
        System.out.println("Meta 1 txnid: " + txnid1);
        
        // The valid meta page has the higher txnid
        ByteBuffer meta = txnid0 >= txnid1 ? meta0 : meta1;
        
        // Read database header from meta page
        // The root database info is at offset 24 in the meta page
        short rootFlags = meta.getShort(24);
        short rootCalculatedPages = meta.getShort(26);
        long rootCardinality = meta.getLong(32); // Actually txnid at 32
        long rootPages = meta.getLong(40);
        
        System.out.println("\nDatabase root info:");
        System.out.println("  Flags: " + rootFlags);
        System.out.println("  Calculated pages: " + rootCalculatedPages);
        System.out.println("  TxnID: " + txnid0);
        System.out.println("  Root pages: " + rootPages);
        
        // Try to find the root page
        // In LMDB, the root page number is stored differently
        // Let's scan for leaf pages
        
        System.out.println("\n=== Scanning for data pages ===");
        
        int entriesFound = 0;
        int leafPages = 0;
        int branchPages = 0;
        
        for (int pageNum = 2; pageNum < fileSize / pageSize && entriesFound < 100; pageNum++) {
            ByteBuffer page = readPage(channel, pageNum, pageSize);
            int pageType = page.getShort(0) & 0xFFFF;
            
            if (pageType == 0x0005) { // Leaf page
                leafPages++;
                int numEntries = page.getShort(4) & 0xFFFF;
                
                for (int i = 0; i < numEntries && entriesFound < 100; i++) {
                    // Leaf page entry format:
                    // Each entry has a variable-size header followed by key and value
                    int entryOffset = 24 + i * 8; // Base offset for entry pointers
                    
                    if (entryOffset + 8 > pageSize) break;
                    
                    short dataSize = page.getShort(entryOffset);
                    int dataOffset = page.getInt(entryOffset + 2) & 0x7FFFFFFF;
                    
                    if (dataOffset + dataSize > pageSize) continue;
                    
                    // Read the node data
                    byte[] nodeData = new byte[dataSize];
                    page.position(dataOffset);
                    page.get(nodeData);
                    
                    // Parse node: key_len(2) + key + value_len(4) + value
                    if (nodeData.length >= 6) {
                        int keyLen = ((nodeData[0] & 0xFF)) | ((nodeData[1] & 0xFF) << 8);
                        
                        if (keyLen < nodeData.length - 6) {
                            byte[] key = Arrays.copyOfRange(nodeData, 2, 2 + keyLen);
                            int valueLen = ((nodeData[2 + keyLen] & 0xFF)) |
                                          ((nodeData[3 + keyLen] & 0xFF) << 8) |
                                          ((nodeData[4 + keyLen] & 0xFF) << 16) |
                                          ((nodeData[5 + keyLen] & 0xFF) << 24);
                            
                            if (2 + keyLen + 4 + valueLen <= nodeData.length) {
                                byte[] value = Arrays.copyOfRange(nodeData, 2 + keyLen + 4, 
                                    2 + keyLen + 4 + valueLen);
                                
                                entriesFound++;
                                
                                String keyStr = new String(key, StandardCharsets.UTF_8);
                                System.out.println("Entry " + entriesFound + ":");
                                System.out.println("  Key: " + (keyStr.length() > 50 ? keyStr.substring(0, 50) + "..." : keyStr));
                                System.out.println("  Key bytes: " + bytesToHex(key));
                                System.out.println("  Value size: " + valueLen + " bytes");
                                
                                // Check if value is pickle
                                if (value.length > 2 && value[0] == (byte)0x80 && value[1] >= 2 && value[1] <= 5) {
                                    System.out.println("  Value type: Python pickle (protocol " + value[1] + ")");
                                }
                            }
                        }
                    }
                }
            } else if (pageType == 0x0002) { // Branch page
                branchPages++;
            } else if (pageType == 0x0003) { // Overflow page
                System.out.println("Found overflow page at " + pageNum);
            }
        }
        
        System.out.println("\n=== Summary ===");
        System.out.println("Leaf pages: " + leafPages);
        System.out.println("Branch pages: " + branchPages);
        System.out.println("Entries found: " + entriesFound);
    }
    
    static ByteBuffer readPage(FileChannel channel, int pageNum, int pageSize) throws Exception {
        ByteBuffer buf = ByteBuffer.allocate(pageSize);
        channel.read(buf, (long) pageNum * pageSize);
        buf.flip();
        return buf;
    }
    
    static String bytesToHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < Math.min(bytes.length, 30); i++) {
            sb.append(String.format("%02x ", bytes[i]));
        }
        if (bytes.length > 30) sb.append("...");
        return sb.toString();
    }
}
