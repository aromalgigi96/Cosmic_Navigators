import React, { useState } from 'react';
import axios from 'axios';
import { Container, Row, Col, Button, Spinner } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Upload } from 'lucide-react';
import './SpaceExplorationPage.css';

// Update this to your EC2 public IP and port
const backendURL = 'http://18.117.144.228:8080';

function SpaceExplorationPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [annotatedUrl, setAnnotatedUrl] = useState(null);
  const [statusMessage, setStatusMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [threshold, setThreshold] = useState(0.3); // New state for detection threshold

  const handleFileChange = (e) => {
    if (!e.target.files.length) return;
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setAnnotatedUrl(null);
    setStatusMessage("");

    if (selectedFile.type.startsWith('image')) {
      const previewUrl = URL.createObjectURL(selectedFile);
      setFilePreview(previewUrl);
    } else if (selectedFile.type.startsWith('video')) {
      setFilePreview(selectedFile.name);
    }
  };

  const handleThresholdChange = (e) => {
    setThreshold(parseFloat(e.target.value));
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setStatusMessage("");

    const formData = new FormData();
    formData.append('file', file);

    // Build the endpoint URL with threshold query parameter (and pdf flag if needed)
    let uploadURL = `${backendURL}/upload?threshold=${threshold}`;
    if (!file.type.startsWith('video') && annotatedUrl && file.type.startsWith('image')) {
      // This condition is to handle pdf download separately, if required.
      // However, pdf download is triggered by handleDownloadReport, so no need to adjust here.
      // This block is just a placeholder if you want to modify image upload URL further.
    }

    try {
      const response = await axios.post(uploadURL, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: file.type.startsWith('video') ? 'json' : 'blob'
      });

      if (file.type.startsWith('video')) {
        const data = response.data;
        if (data.video_id) {
          // Append the threshold parameter to the streaming URL as well
          setAnnotatedUrl(`${backendURL}/stream_video/${data.video_id}?threshold=${threshold}`);
          setStatusMessage("Video uploaded. Now streaming...");
        } else {
          setStatusMessage("Unexpected response for video upload.");
        }
      } else {
        const blob = new Blob([response.data], { type: "image/jpeg" });
        const objectUrl = URL.createObjectURL(blob);
        setAnnotatedUrl(objectUrl);
        setStatusMessage("Annotated image ready!");
      }
    } catch (error) {
      console.error("Upload failed:", error);
      setStatusMessage("Error uploading file.");
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!file) return;
    setLoading(true);
    setStatusMessage("");

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Include both pdf and threshold parameters in the query
      const response = await axios.post(`${backendURL}/upload?pdf=true&threshold=${threshold}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: 'blob'
      });
      const pdfBlob = new Blob([response.data], { type: 'application/pdf' });
      const downloadUrl = URL.createObjectURL(pdfBlob);
      const a = document.createElement('a');
      a.href = downloadUrl;
      a.download = 'result.pdf';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(downloadUrl);
      setStatusMessage("Report downloaded!");
    } catch (error) {
      console.error("Download failed:", error);
      setStatusMessage("Error downloading report.");
    } finally {
      setLoading(false);
    }
  };

  const isVideo = file && file.type && file.type.startsWith('video');
  const rightCardTitle = isVideo ? "Annotated Video" : "Annotated Image";

  return (
    <div className="explore-page">
      {/* Navigation Header */}
      <header className="explore-nav">
        <Container>
          <div className="nav-content d-flex justify-content-between align-items-center">
            <Button variant="link" className="back-button" onClick={() => navigate('/')}>
              <ArrowLeft size={20} /> Back to Home
            </Button>
            <h2 className="page-title">Space Exploration</h2>
          </div>
        </Container>
      </header>

      {/* Main Content */}
      <Container className="explore-container" fluid>
        <Row className="justify-content-center">
          <Col xs={12} md={10} lg={9}>
            <div className="explore-wrapper">
              {/* Left: Upload Card */}
              <div className="custom-card upload-card">
                <h4 className="card-title text-center mb-4">
                  Upload an image or video to detect &amp; classify with YOLO + ResNet
                </h4>
                <div className="upload-dashed-area">
                  <Upload size={40} className="upload-icon" />
                  <p className="upload-text">Choose File</p>
                  <input type="file" className="upload-input" onChange={handleFileChange} />
                  {file && file.type.startsWith('image') && (
                    <img src={filePreview} alt="Preview" className="file-preview-image" />
                  )}
                  {file && file.type.startsWith('video') && (
                    <p className="file-preview-video">Selected video: {filePreview}</p>
                  )}
                </div>
                {/* Threshold Slider */}
                <div className="mt-3">
                  <label htmlFor="thresholdSlider">
                    Confidence Threshold: {threshold.toFixed(2)}
                  </label>
                  <input
                    id="thresholdSlider"
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={threshold}
                    onChange={handleThresholdChange}
                    style={{ width: "100%" }}
                  />
                </div>
                <div className="d-grid gap-2 mt-3">
                  <Button variant="primary" onClick={handleUpload} disabled={loading || !file}>
                    {loading ? (
                      <>
                        <Spinner animation="border" size="sm" /> Processing...
                      </>
                    ) : (
                      "Upload & Process"
                    )}
                  </Button>
                </div>
                {statusMessage && <p className="status-message mt-3">{statusMessage}</p>}
                {file && !isVideo && annotatedUrl && (
                  <div className="d-grid gap-2 mt-3">
                    <Button variant="secondary" onClick={handleDownloadReport} disabled={loading}>
                      Download Report (PDF)
                    </Button>
                  </div>
                )}
              </div>

              {/* Right: Annotated Result Card */}
              <div className="custom-card annotated-card">
                <h4 className="card-title text-center mb-4">{rightCardTitle}</h4>
                {annotatedUrl ? (
                  <div className="annotated-result-box">
                    {isVideo ? (
                      <img src={annotatedUrl} alt="Annotated Video Stream" className="annotated-video-img" />
                    ) : (
                      <img className="annotated-result" src={annotatedUrl} alt="Annotated" />
                    )}
                  </div>
                ) : (
                  <div className="no-image-box">
                    <p>No image selected</p>
                  </div>
                )}
              </div>
            </div>
          </Col>
        </Row>
      </Container>
    </div>
  );
}

export default SpaceExplorationPage;
