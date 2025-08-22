// This file provides a comprehensive deep analysis tool for visualizing data from Excel files, including word clouds, evidence snippets, hotspot maps,
//  and drilldown capabilities
import React, { useState } from "react";
import "leaflet/dist/leaflet.css";
import { DataGrid } from "@mui/x-data-grid";
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from "recharts";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import {
    Box,
    Typography,
    Button,
    CircularProgress,
    Alert,
    Card,
    CardContent,
    Stack,
    Select,
    MenuItem,
    InputLabel,
    FormControl,
    TextField,
} from "@mui/material";
import Autocomplete from "@mui/material/Autocomplete";
import * as XLSX from "xlsx";

export default function DeepAnalyze() {
    const [file, setFile] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [groupBy, setGroupBy] = useState("city");
    const [groupValue, setGroupValue] = useState("");
    const [cityOptions, setCityOptions] = useState([]);
    const [storeOptions, setStoreOptions] = useState([]);
    const [addressOptions, setAddressOptions] = useState([]);
    const [excelRows, setExcelRows] = useState([]);
    const [selectedCategory, setSelectedCategory] = useState("");
    const [filteredRows, setFilteredRows] = useState([]);


    const onFileChange = (e) => {
        const file = e.target.files[0];
        setFile(file);
        setResult(null);
        setError("");
        setGroupValue("");
        setCityOptions([]);
        setStoreOptions([]);
        setAddressOptions([]);
        if (!file) return;


        const reader = new FileReader();
        reader.onload = (evt) => {
            const data = new Uint8Array(evt.target.result);
            const workbook = XLSX.read(data, { type: "array" });
            const sheetName = workbook.SheetNames[0];
            const worksheet = workbook.Sheets[sheetName];
            const json = XLSX.utils.sheet_to_json(worksheet);
            setExcelRows(json);

            const uniq = (arr) =>
                Array.from(new Set(arr.filter((x) => x && String(x).trim()))).sort();

            setCityOptions(uniq(json.map((row) => row.city)));
            setStoreOptions(uniq(json.map((row) => row.store)));
            setAddressOptions(uniq(json.map((row) => row.address)));
        };
        reader.readAsArrayBuffer(file);
    };


    const groupOptions =
        groupBy === "city"
            ? cityOptions
            : groupBy === "store"
                ? storeOptions
                : addressOptions;


    const fetchAnalysis = async (drilldown = false) => {
        if (!file) {
            setError("Please upload a .xlsx file first.");
            return;
        }
        setLoading(true);
        setError("");
        setResult(null);

        const formData = new FormData();
        formData.append("file", file);
        let url = "/visualize/deep-analysis";
        if (drilldown && groupBy && groupValue) {
            url += `?group_by=${groupBy}&group_value=${encodeURIComponent(groupValue)}`;
        }

        try {
            const res = await fetch(url, { method: "POST", body: formData });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || res.statusText);
            }
            const data = await res.json();
            setResult(data);
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };


    const handleDrilldown = async () => {
        if (!groupValue) {
            setError(`Please select a ${groupBy} value for drilldown.`);
            return;
        }
        fetchAnalysis(true);
    };

    return (
        <Box p={3}>
            <Typography variant="h4" gutterBottom>
                DeepAnalyze: Wordclouds, Evidence, Drilldown
            </Typography>
            <input type="file" accept=".xlsx" onChange={onFileChange} />

            {/* Analysis & Drilldown controls */}
            <Stack direction="row" spacing={2} mt={2}>
                <Button
                    variant="contained"
                    disabled={loading}
                    onClick={() => fetchAnalysis(false)}
                >
                    {loading ? <CircularProgress size={20} /> : "Run Analysis"}
                </Button>
                <FormControl size="small" sx={{ minWidth: 140 }}>
                    <InputLabel id="groupby-label">Group By</InputLabel>
                    <Select
                        labelId="groupby-label"
                        value={groupBy}
                        label="Group By"
                        onChange={e => {
                            setGroupBy(e.target.value);
                            setGroupValue("");
                        }}
                    >
                        <MenuItem value="city">City</MenuItem>
                        <MenuItem value="store">Store</MenuItem>
                        <MenuItem value="address">Address</MenuItem>
                    </Select>
                </FormControl>
                <Autocomplete
                    size="small"
                    sx={{ minWidth: 220 }}
                    options={groupOptions}
                    value={groupValue}
                    onChange={(e, val) => setGroupValue(val || "")}
                    renderInput={(params) => (
                        <TextField
                            {...params}
                            label={`Select ${groupBy.charAt(0).toUpperCase() + groupBy.slice(1)}`}
                            placeholder={`Type or choose ${groupBy}`}
                        />
                    )}
                    freeSolo // allow typing custom value
                />
                <Button
                    variant="outlined"
                    disabled={loading || !groupValue}
                    onClick={handleDrilldown}
                >
                    {loading ? <CircularProgress size={20} /> : "Run Drilldown"}
                </Button>
            </Stack>

            {error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                    {error}
                </Alert>
            )}

            {/* --- Wordcloud Gallery --- */}
            {result?.wordclouds && (
                <Box mt={4}>
                    <Typography variant="h6" gutterBottom>Word Clouds by Category</Typography>
                    <Stack direction="row" flexWrap="wrap" gap={3}>
                        {Object.entries(result.wordclouds).map(([cat, url]) => (
                            <Box key={cat} textAlign="center" mb={2}>
                                <img
                                    src={`/static/${url.split("data/")[1]}`}
                                    alt={cat}
                                    style={{ maxWidth: 240, borderRadius: 10, marginBottom: 8, background: "#fff" }}
                                />
                                <Typography variant="caption">{cat}</Typography>
                            </Box>
                        ))}
                    </Stack>
                </Box>
            )}

            {/* --- Evidence Table per Review --- */}
            {result?.snippet_tables && (
                <Box mt={4}>
                    <Typography variant="h6" gutterBottom>Evidence Snippets per Review</Typography>
                    <Stack spacing={2}>
                        {result.snippet_tables.slice(0, 20).map(({ review_index, evidence }) => (
                            <Card key={review_index} sx={{ my: 1 }}>
                                <CardContent>
                                    <Typography variant="subtitle2" color="primary">
                                        Review #{review_index}
                                    </Typography>
                                    {Object.entries(evidence).map(([cat, sentences]) => (
                                        <Box key={cat} mb={1}>
                                            <Typography variant="caption" color="secondary">{cat}:</Typography>
                                            <ul style={{ margin: 0, paddingLeft: 16 }}>
                                                {sentences.map((sent, i) => <li key={i}>{sent}</li>)}
                                            </ul>
                                        </Box>
                                    ))}
                                </CardContent>
                            </Card>
                        ))}
                    </Stack>
                    {result.snippet_tables.length > 20 && (
                        <Typography variant="caption" color="text.secondary">
                            Showing first 20 reviews only.
                        </Typography>
                    )}
                </Box>
            )}

            {/* --- Dropdown Drilldown Summary --- */}
            {result?.dropdown_summary && (
                <Box mt={4}>
                    <Typography variant="h6" gutterBottom>
                        Drilldown: {result.dropdown_summary.group_by} = {result.dropdown_summary.group_value}
                    </Typography>
                    <Typography>Reviews: {result.dropdown_summary.n_reviews}</Typography>
                    <Typography>Avg Rating: {result.dropdown_summary.avg_rating?.toFixed(2) || "N/A"}</Typography>
                    <Typography>Top Categories: {result.dropdown_summary.top_categories?.join(", ")}</Typography>
                    <Typography variant="subtitle2" mt={1}>Full Distribution:</Typography>
                    <ul>
                        {Object.entries(result.dropdown_summary.category_distribution || {}).map(
                            ([cat, count]) => <li key={cat}>{cat}: {count}</li>
                        )}
                    </ul>
                    <Box mt={2}>
                        <Typography variant="subtitle1">Distribution Chart:</Typography>
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart
                                data={Object.entries(result.dropdown_summary.category_distribution || {}).map(
                                    ([category, count]) => ({ category, count })
                                )}
                                layout="vertical"
                                margin={{ top: 20, right: 40, left: 20, bottom: 20 }}
                            >
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis type="number" />
                                <YAxis type="category" dataKey="category" width={210} />
                                <Tooltip />
                                <Bar
                                    dataKey="count"
                                    fill="#8884d8"
                                    onClick={(data, index) => {
                                        if (!data || !data.category) return;
                                        setSelectedCategory(data.category);
                                        const matches = excelRows.filter(
                                            row =>
                                                row.Top1 === data.category ||
                                                (Array.isArray(row.most_accusations)
                                                    ? row.most_accusations.includes(data.category)
                                                    : typeof row.most_accusations === "string" &&
                                                    row.most_accusations.includes(data.category))
                                        );
                                        setFilteredRows(matches);
                                    }}
                                />
                            </BarChart>
                        </ResponsiveContainer>
                    </Box>
                    {selectedCategory && filteredRows.length > 0 && (
                        <Box mt={3}>
                            <Typography variant="subtitle2" gutterBottom>
                                Reviews for: <b>{selectedCategory}</b>
                            </Typography>
                            <DataGrid
                                rows={filteredRows.map((row, i) => ({ id: i, ...row }))}
                                columns={[
                                    { field: "Translated_review2", headerName: "Review", flex: 2 },
                                    { field: "Top1", headerName: "Top Accusation", flex: 1 },
                                    { field: "rating", headerName: "Rating", flex: 1 },
                                    { field: "city", headerName: "City", flex: 1 },
                                    { field: "store", headerName: "Store", flex: 1 },
                                ]}
                                autoHeight
                                pageSize={10}
                                rowsPerPageOptions={[10, 50]}
                                sx={{ mt: 1, background: "#fff" }}
                            />
                        </Box>
                    )}
                </Box>
            )}

            {/* --- Complaint Hotspots Map --- */}
            {excelRows.length > 0 && (
                <Box mt={6}>
                    <Typography variant="h6" gutterBottom>
                        Complaint Hotspots Map
                    </Typography>
                    <MapContainer
                        center={[51, 10]}
                        zoom={6}
                        style={{ height: 400, width: "100%" }}
                    >
                        <TileLayer
                            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                            attribution="&copy; OpenStreetMap contributors"
                        />
                        {excelRows.map((row, i) =>
                            row.latitude && row.longitude ? (
                                <Marker key={i} position={[Number(row.latitude), Number(row.longitude)]}>
                                    <Popup>
                                        <div>
                                            <b>{row.store || row.city}</b><br />
                                            {row.Translated_review2}<br />
                                            <b>Top1:</b> {row.Top1}
                                        </div>
                                    </Popup>
                                </Marker>
                            ) : null
                        )}
                    </MapContainer>
                </Box>
            )}
        </Box>
    );
}
