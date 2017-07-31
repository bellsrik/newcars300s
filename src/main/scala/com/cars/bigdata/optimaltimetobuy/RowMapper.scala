package com.cars.bigdata.optimaltimetobuy

import java.util.Date

/**
  * Created by sbellary on 5/26/2017.
  */
//SINGLETON OBJECT OTB MAPPER
object RowMapper {

  org.apache.spark.sql.catalyst.encoders.OuterScopes.addOuterScope(this)
  final case class OutputLine(vehicle_id: String, current_date: String, pred_days: String, predicted_date: String)

  //CREATE CASE CLASS FOR INPUT FILE
  case class Row1 ( vehicle_id: String,
                   dma_code: String,
                   mileage: String,
                   price: String,
                   photo_count: String,
                   days: String,
                   vdp_views: String,
                   total_vehicles: String,
                   make_id: String,
                   make_model_id: String,
                   model_year: String,
                   trim_id: String)

  //CREATE MAPPER FUNCTION TO SPLIT LINES AND CREATE ROW BASED ON CASE CLASS ROW
  def trainDataMapper(line: String): Row1 = {
    //SPLIT CSV LINE
    val columns = line.split(",")
    val row : Row1 = Row1( columns(0).trim(),
      columns(2).trim(),
      columns(3).trim(),
      columns(4).trim(),
      columns(5).trim(),
      columns(6).trim(),
      columns(7).trim(),
      columns(8).trim(),
      columns(11).trim(),
      columns(12).trim(),
      columns(13).trim(),
      columns(14).trim())

    //RETURNS ROW
    row
  }
  //CREATE CASE CLASS FOR INPUT FILE
  case class VdRow ( vehicle_id: String,
                    dma_code: String,
                    mileage: String,
                    price: String,
                    photo_count: String,
                    days: String,
                    total_vehicles: String,
                    make_id: String,
                    make_model_id: String,
                    model_year: String,
                    trim_id: String)

  //CREATE MAPPER FUNCTION TO SPLIT LINES AND CREATE ROW BASED ON CASE CLASS ROW
  def vehicleDailyMapper(line: String): VdRow = {
    //SPLIT CSV LINE
    val columns = line.split(",")
    val vdrow : VdRow = VdRow( columns(0).trim(),
      columns(7).trim(),
      columns(22).trim(),
      columns(21).trim(),
      columns(20).trim(),
      columns(27).trim(),
      columns(23).trim(),
      columns(8).trim(),
      columns(9).trim(),
      columns(10).trim(),
      columns(11).trim())

    //RETURNS ROW
    vdrow
  }



}
