import React, { useState, useMemo, useEffect } from "react";
import { Accordion, AccordionSummary, AccordionDetails } from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import {
    Box,
    Typography,
    InputLabel,
    Select,
    MenuItem,
    Button,
    CircularProgress,
    Alert,
    Stack,
    FormControl,
    FormGroup,
    FormControlLabel,
    Checkbox,
    Tooltip as MuiTooltip,
} from "@mui/material";
import * as XLSX from "xlsx";
import { DataGrid } from "@mui/x-data-grid";
import {
    ResponsiveContainer,
    LineChart,
    Line,
    XAxis,
    YAxis,
    Tooltip as ChartTooltip,
    Legend,
    CartesianGrid,
    BarChart,
    Bar,
} from "recharts";


const COLORS = [
    "#8884d8", "#82ca9d", "#ffc658", "#d0ed57",
    "#a4de6c", "#8dd1e1", "#83a6ed", "#8e44ad",
    "#e74c3c", "#f39c12", "#2ecc71", "#e67e22",
];

export default function VisualizationPage() {
    const [file, setFile] = useState(null);
    const [sheetData, setSheetData] = useState(null);
    const [tierResult, setTierResult] = useState(null);
    const [rankResult, setRankResult] = useState(null);
    const [semanticResult, setSemanticResult] = useState(null);
    const [topAccResult, setTopAccResult] = useState(null);
    const [trendResult, setTrendResult] = useState(null);
    const [selectedCategories, setSelectedCategories] = useState([]);
    const [benchmarks, setBenchmarks] = useState(null);
    const [groupBreakdown, setGroupBreakdown] = useState(null);
    const [groupBy, setGroupBy] = useState("city");
    const [accSource, setAccSource] = useState("tier");
    const [activeView, setActiveView] = useState(null);
    const [selectedGroup, setSelectedGroup] = useState("");
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);


    useEffect(() => {
        setSelectedGroup("");
    }, [groupBy]);


    const renderSheet = () => {
        if (!sheetData || !sheetData.length) return null;
        const headers = sheetData[0];
        const rows = sheetData.slice(1).map((row, i) => {
            const obj = { id: i };
            headers.forEach((h, j) => {
                obj[h || `Col${j}`] = row[j];
            });
            return obj;
        });
        const columns = headers.map((h, i) => ({
            field: h || `Col${i}`,
            headerName: h || `Column ${i + 1}`,
            flex: 1,
            sortable: true,
            renderCell: params => (
                <MuiTooltip title={params.value || ""} arrow>
                    <span style={{
                        whiteSpace: "nowrap",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        display: "block"
                    }}>
                        {params.value}
                    </span>
                </MuiTooltip>
            )
        }));

        return (
            <>
                <Typography variant="h6" gutterBottom>📄 Uploaded Excel Preview:</Typography>
                <Box sx={{ height: 500, width: "100%" }}>
                    <DataGrid
                        rows={rows}
                        columns={columns}
                        pageSize={20}
                        rowsPerPageOptions={[20, 50, 100]}
                        checkboxSelection
                        disableRowSelectionOnClick
                    />
                </Box>
            </>
        );
    };

    // filter group-breakdown by selectedGroup
    const plotData = useMemo(() => {
        if (!groupBreakdown) return [];
        if (!selectedGroup) return groupBreakdown.matrix;
        return groupBreakdown.matrix.filter(
            row => row[groupBreakdown.group_col] === selectedGroup
        );
    }, [groupBreakdown, selectedGroup]);

    // file load handler
    const onFileChange = e => {
        const f = e.target.files[0];
        setFile(f);
        setTierResult(null);
        setRankResult(null);
        setSemanticResult(null);
        setTopAccResult(null);
        setTrendResult(null);
        setBenchmarks(null);
        setGroupBreakdown(null);
        setActiveView(null);
        setError("");
        setSheetData(null);

        const reader = new FileReader();
        reader.onload = evt => {
            const data = new Uint8Array(evt.target.result);
            const workbook = XLSX.read(data, { type: "array" });
            const sheetName = workbook.SheetNames[0];
            const worksheet = workbook.Sheets[sheetName];
            const json = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
            setSheetData(json);
        };
        reader.readAsArrayBuffer(f);
    };

    // main visualize handler
    const handleVisualize = async type => {
        if (!file) {
            setError("Upload a cleaned Excel file.");
            return;
        }
        setLoading(true);
        setError("");
        const form = new FormData();
        form.append("file", file);

        let endpoint;
        if (type === "tier") endpoint = "/visualize";
        else if (type === "rank") endpoint = "/visualize/by-rank";
        else if (type === "semantic") endpoint = "/visualize/semantic";
        else if (type === "top-accusations") endpoint = "/visualize/top-accusations";
        else if (type === "trend") endpoint = "/visualize/top1-category-trends";
        else if (type === "benchmarks") endpoint = "/visualize/store-benchmarks";
        else if (type === "grouped") {
            endpoint = `/visualize/grouped-breakdown?group_by=${groupBy}&source=${accSource}`;
        }

        try {
            const res = await fetch(endpoint, { method: "POST", body: form });
            const json = await res.json();
            if (!res.ok) throw new Error(json.detail || res.statusText);

            // JSON-only views
            if (type === "trend") {
                setTrendResult(json);
                setSelectedCategories(json.categories);
                setActiveView(type);
                return;
            }
            if (type === "benchmarks") {
                setBenchmarks(json.benchmarks);
                setActiveView(type);
                return;
            }
            if (type === "grouped") {
                setGroupBreakdown(json);
                setActiveView(type);
                return;
            }

            // image-based views
            let path = json.plot;
            if (!path.startsWith("/static/")) {
                path = `/static/${path.replace(/^.*data[\\/]/, "")}`;
            }
            const blob = await fetch(path).then(r => {
                if (!r.ok) throw new Error("Image not found");
                return r.blob();
            });
            const url = URL.createObjectURL(blob);

            if (type === "tier") setTierResult({ ...json, imageUrl: url });
            if (type === "rank") setRankResult({ ...json, imageUrl: url });
            if (type === "semantic") setSemanticResult({ ...json, imageUrl: url });
            if (type === "top-accusations") setTopAccResult({ ...json, imageUrl: url });

            setActiveView(type);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <Box p={3}>
            <Typography variant="h4" gutterBottom>📊 Visualization Dashboard</Typography>

            {/* Action Buttons */}
            <Stack direction="row" spacing={2} mb={3}>
                <Button variant="outlined" component="label">
                    Choose File
                    <input type="file" hidden accept=".xlsx" onChange={onFileChange} />
                </Button>
                <Button variant="contained" disabled={loading} onClick={() => handleVisualize("tier")}>
                    {loading && activeView === "tier" ? <CircularProgress size={20} /> : "Run Evidence Breakdown"}
                </Button>
                <Button variant="contained" disabled={loading} onClick={() => handleVisualize("rank")}>
                    {loading && activeView === "rank" ? <CircularProgress size={20} /> : "Run Top-5 Breakdown"}
                </Button>
                <Button variant="contained" disabled={loading} onClick={() => handleVisualize("semantic")}>
                    {loading && activeView === "semantic" ? <CircularProgress size={20} /> : "Run Semantic Heatmap"}
                </Button>
                <Button variant="contained" disabled={loading} onClick={() => handleVisualize("top-accusations")}>
                    {loading && activeView === "top-accusations" ? <CircularProgress size={20} /> : "Run Top-10 Accusations"}
                </Button>
                <Button variant="contained" disabled={loading} onClick={() => handleVisualize("trend")}>
                    {loading && activeView === "trend" ? <CircularProgress size={20} /> : "Run Top-1 Category Trends"}
                </Button>
                <Button variant="contained" disabled={loading} onClick={() => handleVisualize("benchmarks")}>
                    {loading && activeView === "benchmarks" ? <CircularProgress size={20} /> : "Run Store Benchmarks"}
                </Button>
                <Button variant="contained" disabled={loading} onClick={() => handleVisualize("grouped")}>
                    {loading && activeView === "grouped" ? <CircularProgress size={20} /> : "Compare by Group"}
                </Button>
            </Stack>

            {/* Dropdown Filters */}
            <Stack direction="row" spacing={2} mb={3} alignItems="center">
                <FormControl size="small">
                    <InputLabel id="group-by-label">Group By</InputLabel>
                    <Select
                        labelId="group-by-label"
                        value={groupBy}
                        label="Group By"
                        onChange={e => setGroupBy(e.target.value)}
                    >
                        <MenuItem value="city">City</MenuItem>
                        <MenuItem value="store">Store</MenuItem>
                        <MenuItem value="address">Address</MenuItem>
                    </Select>
                </FormControl>
                <FormControl size="small">
                    <InputLabel id="acc-source-label">Accusation Source</InputLabel>
                    <Select
                        labelId="acc-source-label"
                        value={accSource}
                        label="Accusation Source"
                        onChange={e => setAccSource(e.target.value)}
                    >
                        <MenuItem value="tier">Tier-Based</MenuItem>
                        <MenuItem value="topk">Top-K Cosine</MenuItem>
                        <MenuItem value="semantic">Semantic Match</MenuItem>
                    </Select>
                </FormControl>
                {groupBreakdown && (
                    <FormControl size="small">
                        <InputLabel id="group-select-label">Select {groupBreakdown.group_col}</InputLabel>
                        <Select
                            labelId="group-select-label"
                            value={selectedGroup}
                            label={`Select ${groupBreakdown.group_col}`}
                            onChange={e => setSelectedGroup(e.target.value)}
                        >
                            <MenuItem value=""><em>All</em></MenuItem>
                            {groupBreakdown.groups.map(g => (
                                <MenuItem key={g} value={g}>{g}</MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                )}
            </Stack>

            {error && <Alert severity="error">⚠️ {error}</Alert>}

            {/* Image-based results */}
            {activeView === "tier" && tierResult && (
                <Box my={4}>
                    <Typography variant="h6">Evidence Tier Heatmap</Typography>
                    <img src={tierResult.imageUrl} alt="Evidence Tier" style={{ maxWidth: "100%" }} />
                </Box>
            )}
            {activeView === "rank" && rankResult && (
                <Box my={4}>
                    <Typography variant="h6">Top-5 Rank Heatmap</Typography>
                    <img src={rankResult.imageUrl} alt="Rank Heatmap" style={{ maxWidth: "100%" }} />
                </Box>
            )}
            {activeView === "semantic" && semanticResult && (
                <Box my={4}>
                    <Typography variant="h6">Semantic Topic Heatmap</Typography>
                    <img src={semanticResult.imageUrl} alt="Semantic Heatmap" style={{ maxWidth: "100%" }} />
                </Box>
            )}
            {activeView === "top-accusations" && topAccResult && (
                <Box my={4}>
                    <Typography variant="h6">Top 10 Accusations by Presence</Typography>
                    <img src={topAccResult.imageUrl} alt="Top10" style={{ maxWidth: "100%" }} />
                </Box>
            )}

            {/* Trend charts */}
            {activeView === "trend" && trendResult && (
                <>
                    <Typography variant="h6" gutterBottom>Select Categories to Display</Typography>
                    <FormGroup row>
                        {trendResult.categories.map(cat => (
                            <FormControlLabel
                                key={cat}
                                control={
                                    <Checkbox
                                        checked={selectedCategories.includes(cat)}
                                        onChange={() =>
                                            setSelectedCategories(prev =>
                                                prev.includes(cat)
                                                    ? prev.filter(x => x !== cat)
                                                    : [...prev, cat]
                                            )
                                        }
                                    />
                                }
                                label={cat}
                            />
                        ))}
                    </FormGroup>
                    <Box mt={2} height={400}>
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={trendResult.chartData} margin={{ top: 20, right: 30, left: 20, bottom: 50 }}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis
                                    dataKey="month"
                                    angle={-45}
                                    textAnchor="end"
                                    height={60}
                                    tickFormatter={m => {
                                        const [y, mo] = m.split("-").map(Number);
                                        return new Date(y, mo - 1).toLocaleString("default", { month: "short", year: "numeric" });
                                    }}
                                />
                                <YAxis unit="%" />
                                <ChartTooltip />
                                <Legend verticalAlign="top" height={36} />
                                {trendResult.categories.map((cat, idx) => (
                                    selectedCategories.includes(cat) && (
                                        <Line
                                            key={cat}
                                            dataKey={cat}
                                            dot={false}
                                            stroke={COLORS[idx % COLORS.length]}
                                            strokeWidth={2}
                                        />
                                    )
                                ))}
                            </LineChart>
                        </ResponsiveContainer>
                    </Box>
                </>
            )}

            {/* Benchmarks */}
            {activeView === "benchmarks" && benchmarks && (
                <Box mt={4}>
                    <Typography variant="h6" gutterBottom>Store Benchmarks</Typography>
                    <Box height={300}>
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={benchmarks}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="city" angle={-45} textAnchor="end" height={60} />
                                <YAxis />
                                <ChartTooltip />
                                <Legend verticalAlign="top" height={36} />
                                <Bar dataKey="avg_rating" name="Avg Rating" fill="#8884d8" />
                                <Bar dataKey="complaint_diversity" name="Complaint Diversity" fill="#82ca9d" />
                            </BarChart>
                        </ResponsiveContainer>
                    </Box>
                </Box>
            )}

            {/* Compare by Group */}
            {activeView === "grouped" && groupBreakdown && (
                <Box mt={4}>
                    <Typography variant="h6" gutterBottom>
                        Top 3 Accusation Categories by {groupBreakdown.group_col}
                    </Typography>
                    <Box height={400}>
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={plotData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis
                                    dataKey={groupBreakdown.group_col}
                                    angle={-45}
                                    textAnchor="end"
                                    height={80}
                                />
                                <YAxis unit="%" />
                                <ChartTooltip />
                                <Legend />
                                {groupBreakdown.categories.map((cat, idx) => (
                                    <Bar
                                        key={cat}
                                        dataKey={cat}
                                        stackId="a"
                                        fill={COLORS[idx % COLORS.length]}
                                    />
                                ))}
                            </BarChart>
                        </ResponsiveContainer>
                    </Box>
                </Box>
            )}

            {/* Fallback sheet preview */}
            {/*!activeView && renderSheet()*/}

            <Accordion sx={{ mt: 4 }}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="subtitle1">Show Excel Sheet Preview</Typography>
                </AccordionSummary>
                <AccordionDetails>
                    {renderSheet()}
                </AccordionDetails>
            </Accordion>
        </Box>
    );
}
