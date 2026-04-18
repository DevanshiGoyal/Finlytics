import { NextRequest, NextResponse } from "next/server";

import { callPythonBridge } from "@/lib/python-bridge";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const search = request.nextUrl.searchParams;
    const datasetPath = search.get("datasetPath") || undefined;
    const k = search.get("k") ? Number(search.get("k")) : undefined;
    const topN = search.get("topN") ? Number(search.get("topN")) : undefined;

    const payload: Record<string, unknown> = {};
    if (datasetPath) {
      payload.datasetPath = datasetPath;
    }
    if (typeof k === "number" && Number.isFinite(k)) {
      payload.k = k;
    }
    if (typeof topN === "number" && Number.isFinite(topN)) {
      payload.topN = topN;
    }

    const data = await callPythonBridge("wealth_persona_insights", payload);
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "wealth persona insights failed" },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const payload = await request.json();
    const data = await callPythonBridge("wealth_persona_insights", payload ?? {});
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "wealth persona insights failed" },
      { status: 500 }
    );
  }
}
