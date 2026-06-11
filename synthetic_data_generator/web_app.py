#!/usr/bin/env python3
"""Simple web interface for the synthetic data generator."""

import io
import json
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file, Response
import pandas as pd

from generator.core import SyntheticDataGenerator

app = Flask(__name__, template_folder="templates", static_folder="static")

CONFIG_PATH = Path("configs/default.yaml")
DATA_DIR = Path("data")


@app.route("/")
def index():
    """Main page with generator form."""
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """Generate synthetic dataset and return preview."""
    try:
        n_samples = int(request.form.get("n_samples", 1000))
        seed = request.form.get("seed", "")
        seed = int(seed) if seed.strip() else None
        labeling_mode = request.form.get("labeling_mode", "rule")

        n_samples = max(10, min(n_samples, 100000))

        generator = SyntheticDataGenerator(
            config_path=CONFIG_PATH,
            n_samples=n_samples,
            seed=seed,
            labeling_mode=labeling_mode,
        )
        df = generator.generate()

        # Save the dataset using the core generator method
        output_path = generator.save(df)

        app.config["last_df"] = df
        app.config["last_info"] = generator.get_info(df)

        preview_html = df.head(50).to_html(
            classes="table table-striped table-hover table-sm",
            index=False,
            border=0,
        )

        info = app.config["last_info"]
        class_balance = info.get("class_balance", {})

        return jsonify({
            "success": True,
            "preview": preview_html,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "positive_rate": round(class_balance.get("positive_rate", 0) * 100, 2),
            "count_true": class_balance.get("count_true", 0),
            "count_false": class_balance.get("count_false", 0),
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/download/<fmt>")
def download(fmt: str):
    """Download generated dataset in specified format."""
    df = app.config.get("last_df")
    if df is None:
        return jsonify({"error": "No dataset generated yet"}), 400

    if fmt == "csv":
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return Response(
            buffer.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=synthetic_data.csv"},
        )
    elif fmt == "json":
        buffer = io.StringIO()
        df.to_json(buffer, orient="records", force_ascii=False, indent=2)
        buffer.seek(0)
        return Response(
            buffer.getvalue(),
            mimetype="application/json",
            headers={"Content-Disposition": "attachment;filename=synthetic_data.json"},
        )
    elif fmt == "parquet":
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        return send_file(
            buffer,
            mimetype="application/octet-stream",
            as_attachment=True,
            download_name="synthetic_data.parquet",
        )
    else:
        return jsonify({"error": f"Unknown format: {fmt}"}), 400


@app.route("/stats")
def stats():
    """Return statistics for the last generated dataset."""
    info = app.config.get("last_info")
    if info is None:
        return jsonify({"error": "No dataset generated yet"}), 400
    return jsonify(info)


@app.route("/datasets")
def list_datasets():
    """List saved datasets in data/ folder."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for f in DATA_DIR.glob("*.csv"):
        files.append({"name": f.name, "size": f.stat().st_size, "format": "csv"})
    for f in DATA_DIR.glob("*.parquet"):
        files.append({"name": f.name, "size": f.stat().st_size, "format": "parquet"})
    return jsonify(files)


@app.route("/validate", methods=["POST"])
def validate_dataset():
    """Validate an existing dataset."""
    try:
        filename = request.json.get("filename")
        if not filename:
            return jsonify({"success": False, "error": "Filename missing"}), 400
        
        filepath = DATA_DIR / filename
        if not filepath.exists():
            return jsonify({"success": False, "error": "File not found"}), 404
            
        if filename.endswith(".csv"):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_parquet(filepath)
            
        generator = SyntheticDataGenerator(config_path=CONFIG_PATH)
        validation_results = generator.validate(df)
        
        # Convert numpy types to native Python types for JSON serialization
        safe_results = json.loads(json.dumps(validation_results, default=str))
        
        return jsonify({
            "success": True,
            "results": safe_results
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/info", methods=["POST"])
def dataset_info():
    """Get info for an existing dataset."""
    try:
        filename = request.json.get("filename")
        if not filename:
            return jsonify({"success": False, "error": "Filename missing"}), 400
            
        filepath = DATA_DIR / filename
        if not filepath.exists():
            return jsonify({"success": False, "error": "File not found"}), 404
            
        if filename.endswith(".csv"):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_parquet(filepath)
            
        generator = SyntheticDataGenerator(config_path=CONFIG_PATH)
        info = generator.get_info(df)
        
        # Convert numpy types to native Python types for JSON serialization
        safe_info = json.loads(json.dumps(info, default=str))
        
        return jsonify({
            "success": True,
            "info": safe_info
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/clear", methods=["POST"])
def clear_datasets():
    """Clear all saved datasets in data/ folder."""
    try:
        deleted = 0
        if DATA_DIR.exists():
            for f in DATA_DIR.glob("*.*"):
                f.unlink()
                deleted += 1
        return jsonify({"success": True, "deleted": deleted})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
