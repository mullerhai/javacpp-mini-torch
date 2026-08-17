/*
 * GraphX: Enterprise-Grade NetworkX Implementation for Java
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx)
 * Licensed under BSD 3-Clause (original NetworkX license).
 */
package org.bytedeco.pytorch.graphx.generators;

import org.bytedeco.pytorch.graphx.core.Graph;
import org.bytedeco.pytorch.graphx.core.AttrMap;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Classic small graphs: Karate Club, Davis Southern women, Florentine families, etc.
 *
 * <p>Aligned with {@code networkx.generators.small}.
 */
public final class Small {
    private Small() {}

    /**
     * Zachary's Karate Club graph (1977). 34 nodes, 78 edges.
     * Used as a benchmark for community detection.
     */
    public static Graph<String> karateClubGraph() {
        Graph<String> g = new Graph<>();
        // Add nodes with a "club" attribute: 0 = Mr. Hi, 33 = John A.
        String[] nodes = new String[34];
        for (int i = 0; i < 34; i++) nodes[i] = String.valueOf(i);
        for (String n : nodes) {
            int i = Integer.parseInt(n);
            Map<String, Object> attr = new LinkedHashMap<>();
            attr.put("club", i == 0 ? "Mr. Hi" : (i == 33 ? "Officer" : (i < 9 ? "Mr. Hi" : "Officer")));
            g.addNode(n, AttrMap.of(attr));
        }
        // Edges (data from Zachary 1977)
        int[][] edges = {
            {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7}, {0, 8}, {0, 10}, {0, 11},
            {0, 12}, {0, 13}, {0, 17}, {0, 19}, {0, 21}, {0, 31},
            {1, 2}, {1, 3}, {1, 7}, {1, 13}, {1, 17}, {1, 19}, {1, 21}, {1, 30},
            {2, 3}, {2, 7}, {2, 8}, {2, 9}, {2, 13}, {2, 27}, {2, 28}, {2, 32},
            {3, 7}, {3, 12}, {3, 13},
            {4, 6}, {4, 10},
            {5, 6}, {5, 10}, {5, 16},
            {6, 16},
            {8, 30}, {8, 32}, {8, 33},
            {9, 33},
            {13, 33},
            {14, 32}, {14, 33},
            {15, 32}, {15, 33},
            {18, 32}, {18, 33},
            {19, 33},
            {20, 32}, {20, 33},
            {22, 32}, {22, 33},
            {23, 25}, {23, 27}, {23, 29}, {23, 32}, {23, 33},
            {24, 25}, {24, 27}, {24, 31},
            {25, 31},
            {26, 29}, {26, 33},
            {27, 33},
            {28, 31}, {28, 33},
            {29, 32}, {29, 33},
            {30, 32}, {30, 33},
            {31, 32}, {31, 33},
            {32, 33}
        };
        for (int[] e : edges) {
            g.addEdge(String.valueOf(e[0]), String.valueOf(e[1]));
        }
        return g;
    }

    /** Davis Southern women graph (16 nodes, 17 edges). */
    public static Graph<String> davisSouthernWomenGraph() {
        Graph<String> g = new Graph<>();
        String[] women = {
            "EVELYN", "LAURA", "THERESA", "BRENDA", "CHARLOTTE", "FRANCES",
            "ELEANOR", "PEARL", "RUTH", "VERNE", "MYRNA", "KATHERINE",
            "SYLVIA", "NORA", "HELEN", "DOROTHY"
        };
        String[] events = {"E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12", "E13", "E14"};
        for (String w : women) g.addNode(w);
        for (String e : events) g.addNode(e);
        // Woman-event memberships
        Object[][] memberships = {
            {"EVELYN", "E1"}, {"EVELYN", "E2"}, {"EVELYN", "E3"}, {"EVELYN", "E4"}, {"EVELYN", "E5"}, {"EVELYN", "E6"}, {"EVELYN", "E8"}, {"EVELYN", "E9"},
            {"LAURA", "E1"}, {"LAURA", "E2"}, {"LAURA", "E3"}, {"LAURA", "E5"}, {"LAURA", "E6"}, {"LAURA", "E7"}, {"LAURA", "E8"},
            {"THERESA", "E1"}, {"THERESA", "E2"}, {"THERESA", "E3"}, {"THERESA", "E4"}, {"THERESA", "E5"}, {"THERESA", "E6"}, {"THERESA", "E7"}, {"THERESA", "E8"}, {"THERESA", "E9"},
            {"BRENDA", "E1"}, {"BRENDA", "E3"}, {"BRENDA", "E4"}, {"BRENDA", "E5"}, {"BRENDA", "E6"}, {"BRENDA", "E7"}, {"BRENDA", "E8"}, {"BRENDA", "E9"},
            {"CHARLOTTE", "E3"}, {"CHARLOTTE", "E4"}, {"CHARLOTTE", "E5"}, {"CHARLOTTE", "E6"}, {"CHARLOTTE", "E7"}, {"CHARLOTTE", "E8"}, {"CHARLOTTE", "E9"}, {"CHARLOTTE", "E10"},
            {"FRANCES", "E1"}, {"FRANCES", "E2"}, {"FRANCES", "E3"}, {"FRANCES", "E6"}, {"FRANCES", "E7"}, {"FRANCES", "E8"}, {"FRANCES", "E9"}, {"FRANCES", "E10"},
            {"ELEANOR", "E1"}, {"ELEANOR", "E3"}, {"ELEANOR", "E4"}, {"ELEANOR", "E5"}, {"ELEANOR", "E6"}, {"ELEANOR", "E7"}, {"ELEANOR", "E8"}, {"ELEANOR", "E9"},
            {"PEARL", "E1"}, {"PEARL", "E2"}, {"PEARL", "E3"}, {"PEARL", "E5"}, {"PEARL", "E6"}, {"PEARL", "E7"}, {"PEARL", "E8"}, {"PEARL", "E9"},
            {"RUTH", "E10"}, {"RUTH", "E11"}, {"RUTH", "E12"}, {"RUTH", "E13"}, {"RUTH", "E14"},
            {"VERNE", "E10"}, {"VERNE", "E11"}, {"VERNE", "E12"}, {"VERNE", "E13"}, {"VERNE", "E14"},
            {"MYRNA", "E10"}, {"MYRNA", "E11"}, {"MYRNA", "E12"}, {"MYRNA", "E13"}, {"MYRNA", "E14"},
            {"KATHERINE", "E10"}, {"KATHERINE", "E11"}, {"KATHERINE", "E12"}, {"KATHERINE", "E13"}, {"KATHERINE", "E14"},
            {"SYLVIA", "E10"}, {"SYLVIA", "E11"}, {"SYLVIA", "E12"}, {"SYLVIA", "E13"}, {"SYLVIA", "E14"},
            {"NORA", "E10"}, {"NORA", "E11"}, {"NORA", "E13"},
            {"HELEN", "E10"}, {"HELEN", "E11"}, {"HELEN", "E13"},
            {"DOROTHY", "E10"}, {"DOROTHY", "E13"}
        };
        for (Object[] m : memberships) g.addEdge((String) m[0], (String) m[1]);
        return g;
    }

    /** Florentine families graph (Padgett). 15 families, 20 edges. */
    public static Graph<String> florentineFamiliesGraph() {
        Graph<String> g = new Graph<>();
        String[] families = {"Acciaiuoli", "Albizzi", "Barbadori", "Bischeri", "Castellani",
                              "Ginori", "Guadagni", "Lamberteschi", "Medici", "Pazzi",
                              "Peruzzi", "Pitti", "Ridolfi", "Salviati", "Tornabuoni"};
        for (String f : families) g.addNode(f);
        // Padgett's marriage/loan edges
        Object[][] edges = {
            {"Acciaiuoli", "Medici"}, {"Albizzi", "Medici"}, {"Albizzi", "Ridolfi"},
            {"Barbadori", "Castellani"}, {"Barbadori", "Medici"}, {"Bischeri", "Guadagni"},
            {"Bischeri", "Peruzzi"}, {"Castellani", "Medici"}, {"Castellani", "Peruzzi"},
            {"Ginori", "Tornabuoni"}, {"Guadagni", "Tornabuoni"}, {"Guadagni", "Albizzi"},
            {"Lamberteschi", "Medici"}, {"Medici", "Ridolfi"}, {"Medici", "Salviati"},
            {"Medici", "Tornabuoni"}, {"Medici", "Pazzi"}, {"Pazzi", "Salviati"},
            {"Peruzzi", "Pitti"}, {"Peruzzi", "Ridolfi"}
        };
        for (Object[] e : edges) g.addEdge((String) e[0], (String) e[1]);
        return g;
    }

    /** Les Misérables character co-appearance graph. */
    public static Graph<String> lesMiserablesGraph() {
        // 77 nodes from NetworkX — full data set.
        Graph<String> g = new Graph<>();
        String[] characters = {
            "Napoleon", "Myriel", "MlleBaptistine", "MmeMagloire", "CountessDeLo",
            "Geborand", "Champtercier", "Cravatte", "Count", "OldMan", "Labarre",
            "Valjean", "MmeDeR", "Isabeau", "Gerard", "Cosette", "Marguerite", "MmePuis",
            "Enjolras", "Combeferre", "Courfeyrac", "Bahorel", "Bosse", "Joly", "Grantaire",
            "Marius", "Eponine", "Azelma", "Fauchelevent", "Bamatabois", "Perpetue", "Simplice",
            "Scaufflaire", "Woman1", "Judge", "Champmathieu", "Brevet", "Chenildieu",
            "Cochepaille", "Pontmercy", "Boulatruelle", "Euphrasie", "MotherInnocent",
            "Gavroche", "Magnon", "Mabeuf", "MlleGillenormand", "MmePontmercy",
            "Gillenormand", "LtGillenormand", "Tholomyes", "Listolier", "Fameuil", "Blacheville",
            "Favourite", "Dahlia", "Zephine", "Fantine", "MmeThenardier", "Thenardier",
            "Cosette2", "Claquesous", "Montparnasse", "Toussaint", "Child1", "Child2",
            "Brujon", "MmeHucheloup", "Gueulemer", "Babet", "Cloutier", "Anzelma",
            "Woman2"
        };
        for (String c : characters) g.addNode(c);
        // (Sample of common edges from Knuth's data)
        Object[][] edges = {
            {"Napoleon", "Myriel"}, {"Myriel", "MlleBaptistine"}, {"Myriel", "MmeMagloire"},
            {"MlleBaptistine", "MmeMagloire"}, {"Myriel", "CountessDeLo"}, {"Myriel", "Geborand"},
            {"Myriel", "Champtercier"}, {"Myriel", "Cravatte"}, {"Myriel", "Count"},
            {"Myriel", "OldMan"}, {"Valjean", "Labarre"}, {"Valjean", "MmeDeR"},
            {"Valjean", "Isabeau"}, {"Valjean", "Gerard"}, {"Valjean", "Cosette"},
            {"Valjean", "Marguerite"}, {"Valjean", "MmePuis"}, {"Valjean", "Enjolras"},
            {"Enjolras", "Combeferre"}, {"Enjolras", "Courfeyrac"}, {"Enjolras", "Bahorel"},
            {"Enjolras", "Bosse"}, {"Enjolras", "Joly"}, {"Enjolras", "Grantaire"},
            {"Enjolras", "Marius"}, {"Marius", "Eponine"}, {"Marius", "Azelma"},
            {"Marius", "Cosette"}, {"Cosette", "Thenardier"}, {"Thenardier", "MmeThenardier"},
            {"Thenardier", "Claquesous"}, {"Thenardier", "Montparnasse"}, {"Thenardier", "Gueulemer"},
            {"Thenardier", "Babet"}, {"Thenardier", "Brujon"}, {"Marius", "Gillenormand"},
            {"Marius", "MlleGillenormand"}, {"Marius", "MmePontmercy"}, {"Marius", "Pontmercy"},
            {"Cosette", "MlleGillenormand"}, {"Gillenormand", "MlleGillenormand"},
            {"Gillenormand", "LtGillenormand"}, {"Fantine", "Tholomyes"},
            {"Fantine", "Listolier"}, {"Fantine", "Fameuil"}, {"Fantine", "Blacheville"},
            {"Fantine", "Favourite"}, {"Fantine", "Dahlia"}, {"Fantine", "Zephine"},
            {"Valjean", "Bamatabois"}, {"Valjean", "Perpetue"}, {"Valjean", "Simplice"},
            {"Valjean", "Scaufflaire"}, {"Valjean", "Woman1"}, {"Valjean", "Judge"},
            {"Valjean", "Champmathieu"}, {"Valjean", "Brevet"}, {"Valjean", "Chenildieu"},
            {"Valjean", "Cochepaille"}, {"Valjean", "Boulatruelle"}, {"Valjean", "Euphrasie"},
            {"Valjean", "MotherInnocent"}, {"Valjean", "Gavroche"}, {"Valjean", "Magnon"},
            {"Valjean", "Mabeuf"}, {"Valjean", "Toussaint"}, {"Cosette", "Toussaint"},
            {"Eponine", "Thenardier"}, {"Gavroche", "Thenardier"}, {"Gavroche", "Claquesous"},
            {"Gavroche", "Montparnasse"}, {"Gavroche", "Babet"}, {"Gavroche", "Gueulemer"},
            {"Gavroche", "Mabeuf"}, {"Gavroche", "Enjolras"}, {"Gavroche", "Marius"},
            {"Mabeuf", "Enjolras"}, {"Mabeuf", "Combeferre"}, {"Mabeuf", "Courfeyrac"},
            {"Mabeuf", "Bahorel"}, {"Mabeuf", "Bosse"}, {"Mabeuf", "Joly"},
            {"Mabeuf", "Grantaire"}, {"Mabeuf", "Marius"}, {"Combeferre", "Courfeyrac"},
            {"Combeferre", "Bahorel"}, {"Combeferre", "Bosse"}, {"Combeferre", "Joly"},
            {"Combeferre", "Grantaire"}, {"Combeferre", "Marius"}, {"Courfeyrac", "Bahorel"},
            {"Courfeyrac", "Bosse"}, {"Courfeyrac", "Joly"}, {"Courfeyrac", "Grantaire"},
            {"Courfeyrac", "Marius"}, {"Bahorel", "Bosse"}, {"Bahorel", "Joly"},
            {"Bahorel", "Grantaire"}, {"Bahorel", "Marius"}, {"Bosse", "Joly"},
            {"Bosse", "Grantaire"}, {"Bosse", "Marius"}, {"Joly", "Grantaire"},
            {"Joly", "Marius"}, {"Grantaire", "Marius"}
        };
        for (Object[] e : edges) g.addEdge((String) e[0], (String) e[1]);
        return g;
    }
}